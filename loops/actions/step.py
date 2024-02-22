"""
Detials
"""
# imports
import torch
from torch.cuda.amp import autocast
from torchmetrics.detection import MeanAveragePrecision
import gc

# class
class Step():
    """ Detials """
    def __init__(self, cfg):
        """ Detials """
        self.cfg = cfg
        self.model_name = self.cfg["model_name"]

        # for multi task
        self.iter_init_flag = False

        # for pseudo labeling
        self.burn_in = False
        self.burn_in_targ_needed = True
    
    def action(self):
        self.action_map = {
            "mask_rcnn": self._instance_seg_action,
            "dual_mask_multi_task": self._multitask_action,
            "polite_teacher_mask_rcnn": self._pseudo_action
        }
        return self.action_map[self.model_name]

    def _instance_seg_action(self, model, train_loader, val_loader, loss, optimiser, device, grad_acc, epoch, log, iter_count, logger):
        """ Detials """
        def train(model, loader, loss_fun, optimiser, device, scaler, grad_acc, epoch, log, iter_count, logger, AWL_flag):
            """ Detials """
            # loop execution setup
            model.train()
            pf_loss = 0
            loss_acc = 0

            # loop
            for i, data in enumerate(loader):
                # get batch
                input, target = data
                input = list(image.to(device) for image in input)
                target = [{k: v.to(device) for k, v in t.items()} for t in target]

                with autocast():
                    output = model.forward(input, target)
                    if AWL_flag:
                        loss = loss_fun(output["loss_classifier"], output["loss_box_reg"], output["loss_mask"], output["loss_objectness"], output["loss_rpn_box_reg"])
                    else:
                        loss = sum(loss for loss in output.values())

                loss = loss/grad_acc
            
                scaler.scale(loss).backward()
                if grad_acc:
                    if (i+1) % grad_acc == 0:
                        scaler.step(optimiser)
                        scaler.update()
                        optimiser.zero_grad()
                else:
                    scaler.step(optimiser)
                    scaler.update()
                    optimiser.zero_grad()

                # recording
                loss_val = loss.item()
                pf_loss += loss_val
                loss_acc += loss_val

                # reporting
                pf_loss = logger.train_loop_reporter(epoch, iter_count, device, pf_loss)
                iter_count += 1

            # logging goes here
            log["epochs"].append(epoch)
            log["train_loss"].append(loss_acc/len(loader))

            # returns

        def validate(model, loader, loss_fun, device , epoch, log, logger):
            """ Detials """
            # loop execution setup
            model.train()
            loss_acc = 0

            for i, data in enumerate(loader):
                # get batch
                input, target = data
                input = list(image.to(device) for image in input)
                target = [{k: v.to(device) for k, v in t.items()} for t in target]

                with torch.no_grad():
                    output = model.forward(input, target)
                    loss = sum(loss for loss in output.values())

                loss_acc += loss.item()

            # set form map evaluation
            model.eval()
            metric = MeanAveragePrecision(iou_type = "segm")
            sup_iter = iter(loader)

            for i in range(len(loader)):
                input, target = next(sup_iter)
                input = list(image.to(device) for image in input)
     
                with torch.autocast("cuda"):
                    with torch.no_grad():
                        predictions = model(input)

                masks_in = predictions[0]["masks"].detach().cpu()
                masks_in = masks_in > 0.5
                masks_in = masks_in.squeeze(1) 
                targs_masks = target[0]["masks"].bool()
                targs_masks = targs_masks.squeeze(1)  
                preds = [dict(masks=masks_in, scores=predictions[0]["scores"].detach().cpu(), labels=predictions[0]["labels"].detach().cpu(),)]
                targs = [dict(masks=targs_masks, labels=target[0]["labels"],)]
                metric.update(preds, targs)

                del predictions, input, target, masks_in, targs_masks, preds, targs
                torch.cuda.empty_cache()
            
            res = metric.compute()
            map = res["map"].item()

            loss = loss_acc/len(loader)
            logger.val_loop_reporter(epoch, device, loss)
            log["val_loss"].append(loss)
            log["map"].append(map)
                
        # initial params
        banner = "--------------------------------------------------------------------------------"
        train_title = "Training"
        val_title = "Validating"

        if isinstance(loss, list):
            loss = loss[1]
            loss.to(device)
            AWL_flag = True
        else:
            AWL_flag = False

        scaler = torch.cuda.amp.GradScaler(enabled=True)

        print(banner)
        print(train_title)
        print(banner)

        train(model, train_loader, loss, optimiser, device, scaler, grad_acc, epoch, log, iter_count, logger, AWL_flag)

        print(banner)
        print(val_title)
        print(banner)

        validate(model, val_loader, loss, device, epoch, log, logger) 
    
    def _multitask_action(self, model, train_loader, val_loader, loss, optimiser, device, grad_acc, epoch, log, iter_count, logger):
        """ Detials """
        def train(model, loader, loss_fun, optimiser, device, scaler, grad_acc, epoch, log, iter_count, logger):
            """ Detials """
            # loop execution setup
            model.train()

            # supervised and self supervised loader extraction
            sup_iter = iter(loader[0])

            # losses ! NEEDS TO BE ADDRESSED
            awl = loss_fun[0]

            primary_grad = grad_acc[0] if grad_acc else 1
            secondar_grad = grad_acc[1] if grad_acc else 1

            sup_loss_acc, ssl_loss_acc, weighted_losses_acc, pf_loss = 0, 0, 0, 0
                
            for i in range(len(loader[0])):
                sup_im, sup_target = next(sup_iter)
                sup_im = list(image.to(device) for image in sup_im)
                sup_target = [{k: v.to(device) for k, v in t.items()} for t in sup_target]

                with autocast():
                    # forward pass
                    sup_output = model.forward(sup_im, sup_target, mode="sup")
                    sup_loss = sum(loss for loss in sup_output.values())

                    sup_loss_acc += sup_loss.item()
                    
                    for i in range(0, secondar_grad):
                        try:
                            ssl_im, ssl_target = next(self.train_ssl_iter) 
                        except StopIteration:
                            print("resetting iter")
                            self.train_ssl_iter = iter(loader[1])
                            ssl_im, ssl_target = next(self.train_ssl_iter)
                        ssl_im = list(image.to(device) for image in ssl_im)
                        ssl_target = [{k: v.to(device) for k, v in t.items()} for t in ssl_target]

                        # forward pass
                        ssl_output = model.forward(sup_im, sup_target, mode="ssl")
                        ssl_loss = sum(loss for loss in ssl_output.values())

                        # in dev equal bs being used. this will need addressing
                        #ssl_loss =+ ssl_loss.div_(secondar_grad)
                    ssl_loss_acc += ssl_loss.item()

                    weighted_loss = awl(sup_output["loss_classifier"], sup_output["loss_box_reg"], sup_output["loss_mask"], sup_output["loss_objectness"], sup_output["loss_rpn_box_reg"],
                                        ssl_output["loss_classifier"], ssl_output["loss_box_reg"], ssl_output["loss_mask"], ssl_output["loss_objectness"], ssl_output["loss_rpn_box_reg"])

                    weighted_losses_acc += weighted_loss.item()
                    pf_loss += weighted_loss.item()
                                
                scaler.scale(weighted_loss).backward()
                if (i+1) % primary_grad == 0:
                    # optimiser step
                    scaler.step(optimiser)
                    scaler.update()
                    optimiser.zero_grad()
                                    
                # reporting
                pf_loss = logger.train_loop_reporter(epoch, iter_count, device, pf_loss)
                iter_count += 1

                # Clear GPU Memory
                del sup_im, sup_target, sup_output, ssl_loss, weighted_loss, ssl_output, ssl_target, ssl_im
                torch.cuda.empty_cache()
                gc.collect()

            # accumulating iter count for ssl iter adjust
            log["iter_accume"] += iter_count*secondar_grad

            # logging
            log["epochs"].append(epoch)
            log["train_loss"].append(weighted_losses_acc/len(loader[0]))
            log["train_sup_loss"].append(sup_loss_acc/len(loader[0]))
            log["train_ssl_loss"].append(ssl_loss_acc/len(loader[0]))

        def validate(model, loader, loss_fun, device , epoch, log, logger):
            """ Detials """
            # loop execution setup
            # configure model
            model.train()            
            if hasattr(model.backbone.body.layer4, "dropout"):
                p = model.backbone.body.layer4.dropout.p
                model.backbone.body.layer4.dropout.p = 0
                
            sup_loss_acc = 0
            ssl_loss_acc = 0
            weighted_losses_acc = 0

            # supervised and self supervised loader extraction
            sup_iter = iter(loader[0])

            # losses
            awl = loss_fun[0]
            _ = loss_fun[0]

            # ssl step adjust goes here
            #ssl_adjust = log["val_it_accume"] % len(loader[1])
            #if ssl_adjust:
            #    print("Adjusting by %s steps" %(ssl_adjust))
            #    for i in range(ssl_adjust):
            #        _, _ = next(ssl_iter)

            # ssl step adjust goes here
            for i in range(len(loader[0])):
                sup_im, sup_target = next(sup_iter)
                sup_im = list(image.to(device) for image in sup_im)
                sup_target = [{k: v.to(device) for k, v in t.items()} for t in sup_target]

                try:
                    ssl_im, ssl_target = next(self.val_ssl_iter) 
                except StopIteration:
                    print("resetting iter")
                    self.val_ssl_iter = iter(loader[1])
                    ssl_im, ssl_target = next(self.val_ssl_iter)  
                ssl_im = list(image.to(device) for image in ssl_im)
                ssl_target = [{k: v.to(device) for k, v in t.items()} for t in ssl_target]

                with torch.no_grad():
                    with autocast():
                        # forward pass
                        sup_output = model.forward(sup_im, sup_target, mode="sup")
                        sup_loss = sum(loss for loss in sup_output.values())
                        ssl_output = model.forward(ssl_im, ssl_target, mode="ssl")
                        ssl_loss = sum(loss for loss in ssl_output.values())

                    sup_loss_acc += sup_loss.item()
                    ssl_loss_acc += ssl_loss.item()

                    weighted_loss = awl(sup_output["loss_classifier"], sup_output["loss_box_reg"], sup_output["loss_mask"], sup_output["loss_objectness"], sup_output["loss_rpn_box_reg"],
                                        ssl_output["loss_classifier"], ssl_output["loss_box_reg"], ssl_output["loss_mask"], ssl_output["loss_objectness"], ssl_output["loss_rpn_box_reg"])

                # collecting losses
                weighted_losses_acc += weighted_loss.item()

                # Clear GPU Memory
                del sup_im, sup_target, sup_output, ssl_loss, weighted_loss, ssl_output, ssl_target, ssl_im, sup_loss
                torch.cuda.empty_cache()
                gc.collect()

            # model re config
            if hasattr(model.backbone.body.layer4, "dropout"):
                model.backbone.body.layer4.dropout.p = p

            # set form map evaluation
            model.eval()
            metric = MeanAveragePrecision(iou_type = "segm")
            sup_iter = iter(loader[0])

            for i in range(len(loader[0])):
                input, target = next(sup_iter)
                input = list(image.to(device) for image in input)
        
                with torch.autocast("cuda"):
                    with torch.no_grad():
                        predictions = model(input)

                masks_in = predictions[0]["masks"].detach().cpu()
                masks_in = masks_in > 0.5
                masks_in = masks_in.squeeze(1) 
                targs_masks = target[0]["masks"].bool()
                targs_masks = targs_masks.squeeze(1)  
                preds = [dict(masks=masks_in, scores=predictions[0]["scores"].detach().cpu(), labels=predictions[0]["labels"].detach().cpu(),)]
                targs = [dict(masks=targs_masks, labels=target[0]["labels"],)]
                metric.update(preds, targs)

                del predictions, input, target, masks_in, targs_masks, preds, targs
                torch.cuda.empty_cache()
            
            res = metric.compute()
            map = res["map"].item()

            # adjusting val iter accumulation for ssl step adjust
            log["val_it_accume"] += len(loader[0])

            # logging
            log["val_loss"].append(weighted_losses_acc/len(loader[0]))
            log["val_sup_loss"].append(sup_loss_acc/len(loader[0]))
            log["val_ssl_loss"].append(ssl_loss_acc/len(loader[0]))
            log["map"].append(map)
            
            logger.val_loop_reporter(epoch, device, log["map"][-1])

        # initial params
        banner = "--------------------------------------------------------------------------------"
        train_title = "Training"
        val_title = "Validating"

        loss[0].to(device)

        scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.train_ssl_iter = iter(train_loader[1])
        self.val_ssl_iter = iter(val_loader[1])

        print(banner)
        print(train_title)
        print(banner)

        train(model, train_loader, loss, optimiser, device, scaler, grad_acc, epoch, log, iter_count, logger)

        print(banner)
        print(val_title)
        print(banner)

        validate(model, val_loader, loss, device, epoch, log, logger)

    def _pseudo_action(self, model, train_loader, val_loader, loss, optimiser, device, grad_acc, epoch, log, iter_count, logger):
        """ Detials """
        def train(model, loader, loss_fun, optimiser, device, scaler, grad_acc, epoch, log, iter_count, logger, AWL_flag):
            """ Detials """
            # loop execution setup
            model.train()
            loss_acc = 0
            pf_loss = 0
            label_acc = 0
            unlabeled_acc = 0
            awl = loss_fun

            label_loader = loader[0]

            grad_lab_acc = 0
            grad_unlab_acc = 0
            grad_weighted_acc = 0

            # loop
            for i, data in enumerate(label_loader):
                # get batch
                _, images, targets = data
                pseduo_images, pseudo_targets = get_pseudo_data(model, loader[1], device)
                
                model.train()
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                with autocast():
                    labeled_output = model.forward(images, targets, forward_type="student")
                del images, targets

                pseduo_images = list(image.to(device) for image in pseduo_images)
                pseudo_targets = [{k: v.to(device) for k, v in t.items()} for t in pseudo_targets]
                with autocast():
                    pseudo_output = model.forward(pseduo_images, pseudo_targets, forward_type="student")
                del pseduo_images, pseudo_targets

                with autocast():
                    weighted_loss = awl(labeled_output["loss_classifier"], 
                                        labeled_output["loss_box_reg"], 
                                        labeled_output["loss_mask"], 
                                        labeled_output["loss_objectness"], 
                                        labeled_output["loss_rpn_box_reg"],
                                        pseudo_output["loss_classifier"], 
                                        pseudo_output["loss_box_reg"], 
                                        pseudo_output["loss_mask"], 
                                        pseudo_output["loss_objectness"], 
                                        pseudo_output["loss_rpn_box_reg"])/grad_acc
                    
                grad_lab_acc += sum(loss for loss in labeled_output.values())/grad_acc
                grad_unlab_acc += sum(loss for loss in pseudo_output.values())/grad_acc
                grad_weighted_acc += weighted_loss/grad_acc

                scaler.scale(weighted_loss).backward()
                if (i+1) % grad_acc == 0:
                    scaler.step(optimiser)
                    scaler.update()
                    optimiser.zero_grad()
                
                    # model update logic goes here
                    if self.burn_in:
                        model.ema_update()

                    # collecting loss data
                    label_acc += grad_lab_acc.item()
                    unlabeled_acc += grad_unlab_acc.item()
                    loss_acc += grad_weighted_acc.item()

                    # reset local accumulators
                    grad_lab_acc = 0
                    grad_unlab_acc = 0
                    grad_weighted_acc = 0

                # recording
                pf_loss += weighted_loss.item()

                # reporting
                pf_loss = logger.train_loop_reporter(epoch, iter_count, device, pf_loss)
                iter_count += 1

            # logging goes here
            if self.burn_in:
                log["epochs"].append(epoch)
                log["train_loss"].append(loss_acc/len(loader))
            else:
                log["train_loss"].append(0)
                log["student_train_loss"].append(loss_acc/len(loader))

        def validate(model, val_loader, loss, device, epoch, log, logger):
            """ Detials """
            """ Detials """
            # loop execution setup
            model.train()
            loss_acc = 0

            for data in val_loader:
                # get batch
                images, targets = data
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                with torch.no_grad():
                    if self.burn_in:
                        output = model.forward(images, targets, forward_type="teacher")
                    else:
                        output = model.forward(images, targets, forward_type="student")
                    loss = sum(loss for loss in output.values())

                loss_acc += loss.item()

            # set form map evaluation
            model.eval()
            metric = MeanAveragePrecision(iou_type = "segm")

            for data in val_loader:
                images, targets = data
                images = list(image.to(device) for image in images)
     
                with torch.autocast("cuda"):
                    with torch.no_grad():
                        if self.burn_in:
                            predictions = model.forward(images, forward_type="teacher")
                        else:
                            predictions = model.forward(images, forward_type="student")


                masks_in = predictions[0]["masks"].detach().cpu()
                masks_in = masks_in > 0.5
                masks_in = masks_in.squeeze(1) 
                targs_masks = targets[0]["masks"].bool()
                targs_masks = targs_masks.squeeze(1)  
                preds = [dict(masks=masks_in, scores=predictions[0]["scores"].detach().cpu(), labels=predictions[0]["labels"].detach().cpu(),)]
                targs = [dict(masks=targs_masks, labels=targets[0]["labels"],)]
                metric.update(preds, targs)

                del predictions, images, targets, masks_in, targs_masks, preds, targs
                torch.cuda.empty_cache()
            
            res = metric.compute()
            map = res["map"].item()

            if self.burn_in_targ_needed:
                return map

            loss = loss_acc/len(val_loader)
            logger.val_loop_reporter(epoch, device, loss)

            if self.burn_in:
                log["val_loss"].append(loss)
                log["map"].append(map)
            else:
                log["val_loss"].append(100)
                log["map"].append(0)
                log["student_val_loss"].append(loss)
                log["student_map"].append(map)

            if map >= self.burn_in_target:
                self.burn_in = True

        def get_pseudo_data(model, loader, device):
            """ Detials """
            try:
                images, heavy_images, _ = next(self.pseudo_iter) 
            except StopIteration:
                print("resetting iter")
                self.train_ssl_iter = iter(loader)
                images, heavy_images = next(self.pseudo_iter)
            image = list(image.to(device) for image in images)

            model.eval()
            with torch.no_grad():
                predictions = model(image, forward_type="teacher")
            outputs = filter_predictions(predictions)

            masks = outputs["masks"]
            area = masks.squeeze(1).sum(dim=[1, 2])

            # look out for this - its temp
            image_id = torch.arange(masks.size(0))

            targets = {
                "boxes": outputs["boxes"],
                "labels": outputs["labels"],
                "masks": masks,
                "image_id": image_id,
                "area": area,
                "iscrowd": torch.zeros((outputs["boxes"].size(0),), dtype=torch.int64) # assuming is crowd is none 
            }

            del predictions, outputs, masks, area
            return heavy_images, (targets,)
        
        def filter_predictions(predictions, threshold=0.5):
            """ Filters the predicted masks to get the masks and meta data from the model """
            masks = (predictions[0]["masks"] > 0.9).float()
            valid_indices = torch.nonzero(predictions[0]["scores"] > threshold).squeeze(1)

            masks = masks[valid_indices]
            boxes = predictions[0]["boxes"][valid_indices]
            labels = predictions[0]["labels"][valid_indices]
            scores = predictions[0]["scores"][valid_indices]

            outputs = {
                "masks": masks.detach().cpu(),
                "boxes": boxes.detach().cpu(),
                "labels": labels.detach().cpu(),
                "scores": scores.detach().cpu()
                }
            del predictions, masks, valid_indices, boxes, labels, scores
            return outputs

        # initial params
        banner = "--------------------------------------------------------------------------------"
        first_val = "Getting burn in target"
        train_title = "Training"
        val_title = "Validating"

        loss = loss[0]
        loss.to(device)
        AWL_flag = True

        scaler = torch.cuda.amp.GradScaler(enabled=True)

        if not self.iter_init_flag:
            self.pseudo_iter = iter(train_loader[1])
            self.iter_init_flag = True

        if self.burn_in_targ_needed:

            print(banner)
            print(first_val)
            print(banner)

            self.burn_in_target = validate(model, val_loader, loss, device, epoch, log, logger)
            self.burn_in_targ_needed = False

        print(banner)
        print(train_title)
        print(banner)

        train(model, train_loader, loss, optimiser, device, scaler, grad_acc, epoch, log, iter_count, logger, AWL_flag)

        print(banner)
        print(val_title)
        print(banner)

        validate(model, val_loader, loss, device, epoch, log, logger) 