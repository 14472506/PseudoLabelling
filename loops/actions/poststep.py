"""
Detials
"""
# imports

# class
class PostStep():
    """ Detials """
    def __init__(self, cfg):
        """ Detials """
        self.cfg = cfg
        self.model_name = self.cfg["model_name"]
        self.stepped = False
    
    def action(self):
        self.action_map = {
            "mask_rcnn": self._instance_seg_action,
            "dual_mask_multi_task": self._multitask_action,
            "polite_teacher_mask_rcnn": self._pseudo_action
        }
        return self.action_map[self.model_name]
    
    def _pseudo_action(self, epoch, model, optimiser, scheduler, logs, logger):
        """ Detials """
        logger.save_model(epoch, model, optimiser, "last")

        #if logs["val_sup_loss"][-1] <= logger.best[0]:
        if logs["val_loss"][-1] <= logger.best[0]:
            if self.stepped:
                logs["post_best_val"].append(logs["val_loss"][-1])
                logs["post_best_epoch"].append(epoch)
                logger.save_model(epoch, model, optimiser, "val_post")
                logger.best[0] = logs["val_sup_loss"][-1]
            else:
                logs["pre_best_val"].append(logs["val_loss"][-1])
                logs["pre_best_epoch"].append(epoch)
                logger.save_model(epoch, model, optimiser, "val_pre")
                logger.best[0] = logs["val_loss"][-1]
        
        if logs["map"][-1] >= logger.best[1]:
            if self.stepped:
                logs["post_best_map"].append(logs["map"][-1])
                logs["post_best_map_epoch"].append(epoch)
                logger.save_model(epoch, model, optimiser, "post")
                logger.best[1] = logs["map"][-1]
            else:
                logs["pre_best_map"].append(logs["map"][-1])
                logs["pre_best_map_epoch"].append(epoch)
                logger.save_model(epoch, model, optimiser, "pre")
                logger.best[1] = logs["map"][-1]

        logger.update_log_file(logs)

        if scheduler:
            self._handle_scheduler_step(scheduler, epoch, model, logger)

    def _multitask_action(self, epoch, model, optimiser, scheduler, logs, logger):
        """ Detials """
        logger.save_model(epoch, model, optimiser, "last")

        #if logs["val_sup_loss"][-1] <= logger.best[0]:
        if logs["val_sup_loss"][-1] <= logger.best[0]:
            if self.stepped:
                logs["post_best_val"].append(logs["val_sup_loss"][-1])
                logs["post_best_epoch"].append(epoch)
                logger.save_model(epoch, model, optimiser, "val_post")
                logger.best[0] = logs["val_sup_loss"][-1]
            else:
                logs["pre_best_val"].append(logs["val_sup_loss"][-1])
                logs["pre_best_epoch"].append(epoch)
                logger.save_model(epoch, model, optimiser, "val_pre")
                logger.best[0] = logs["val_sup_loss"][-1]
        
        if logs["map"][-1] >= logger.best[1]:
            if self.stepped:
                logs["post_best_map"].append(logs["map"][-1])
                logs["post_best_map_epoch"].append(epoch)
                logger.save_model(epoch, model, optimiser, "post")
                logger.best[1] = logs["map"][-1]
            else:
                logs["pre_best_map"].append(logs["map"][-1])
                logs["pre_best_map_epoch"].append(epoch)
                logger.save_model(epoch, model, optimiser, "pre")
                logger.best[1] = logs["map"][-1]

        logger.update_log_file(logs)

        if scheduler:
            self._handle_scheduler_step(scheduler, epoch, model, logger)

    def _instance_seg_action(self, epoch, model, optimiser, scheduler, logs, logger):
        """ Detials """
        logger.save_model(epoch, model, optimiser, "last")

        if logs["val_loss"][-1] <= logger.best[0]:
            if self.stepped:
                logs["post_best_val"].append(logs["val_loss"][-1])
                logs["post_best_epoch"].append(epoch)
                logger.save_model(epoch, model, optimiser, "val_post")
                logger.best[0] = logs["val_loss"][-1]
            else:
                logs["pre_best_val"].append(logs["val_loss"][-1])
                logs["pre_best_epoch"].append(epoch)
                logger.save_model(epoch, model, optimiser, "val_pre")
                logger.best[0] = logs["val_loss"][-1]
        
        if logs["map"][-1] >= logger.best[1]:
            if self.stepped:
                logs["post_best_map"].append(logs["map"][-1])
                logs["post_best_map_epoch"].append(epoch)
                logger.save_model(epoch, model, optimiser, "post")
                logger.best[1] = logs["map"][-1]
            else:
                logs["pre_best_map"].append(logs["map"][-1])
                logs["pre_best_map_epoch"].append(epoch)
                logger.save_model(epoch, model, optimiser, "pre")
                logger.best[1] = logs["map"][-1]

        logger.update_log_file(logs)

        if scheduler:
            self._handle_scheduler_step(scheduler, epoch, model, logger)
        
    def _handle_scheduler_step(self, scheduler, epoch, model, logger):
        """ Detials """
        if epoch == logger.step-1:
            logger.load_model(model, "pre")

            if isinstance(logger.best, list):
                logger.best = [float('inf'), 0]
            else:
                logger.best = float('inf')

            self.stepped = True
        scheduler.step()