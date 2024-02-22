"""
Detials
"""
# imports
import os 
import json

# funcs
def main(exp_list, models_list, root_dir = "outputs/results", file_name="extended_results.json", ev_type = "avg"):
    """ Details """
    accume_results = {}
    for exp in exp_list:
        avg_map, avg_ap50, avg_ap75 = 0, 0, 0
        pass_flag = False

        for model in models_list:
            targ = os.path.join(root_dir, exp, model, file_name)
            mAP, ap50, ap75 = get_data(targ)
            pass_flag = False
            if ev_type == "avg":
                avg_map += mAP/len(models_list)
                avg_ap50 += ap50/len(models_list)
                avg_ap75 += ap75/len(models_list)
            
        res = [round(avg_map,3), round(avg_ap50,3), round(avg_ap75,3)]
        accume_results[exp] = res
    
    print(accume_results)

def get_data(target):
    """ Detials """
    with open(target, "r") as file:
        data = json.load(file)
    
    pre_map = data["pre_step"]["map"]
    post_map = data["post_step"]["map"]

    if post_map > pre_map:
        return post_map, data["post_step"]["map_50"], data["post_step"]["map_75"]
    return pre_map, data["pre_step"]["map_50"], data["pre_step"]["map_75"]


    
# execution
if __name__ == "__main__":
    exp_list = ["OBA_0perc_MRCNN", "OBA_10perc_MRCNN", "OBA_25perc_MRCNN", "OBA_50perc_MRCNN", "OBA_75perc_MRCNN", "OBA_90perc_MRCNN"]
    models_list = ["model_0", "model_1", "model_2"]
    main(exp_list, models_list, ev_type="avg")
