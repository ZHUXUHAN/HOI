import json
import pickle
import numpy as np
import os

detection = []
# Action_dic = json.load(open('./data/action_index.json'))
# Action_dic_inv = {y: x for x, y in Action_dic.items()}
# print(Action_dic_inv)

action_classes = ['hold', 'stand', 'sit', 'ride', 'walk', 'look', 'hit', 'eat', 'jump', 'lay', 'talk_on_phone', 'carry',
                  'throw', 'catch', 'cut', 'run', 'work_on_computer', 'ski', 'surf', 'skateboard', 'smile', 'drink',
                  'kick', 'point', 'read', 'snowboard']


action_agent = ['hold_agent', 'smile_agent', 'snowboard_agent', 'surf_agent', 'eat_agent', 'jump_agent', 'catch_agent',
                'ski_agent', \
                'skateboard_agent', 'point_agent', 'stand_agent', 'cut_agent', 'work_on_computer_agent', 'lay_agent',
                'drink_agent', \
                'look_agent', 'read_agent', 'run_agent', 'sit_agent', 'hit_agent', 'walk_agent', 'carry_agent',
                'throw_agent', \
                'kick_agent', 'talk_on_phone_agent', 'ride_agent']

action_classes_29 = ['hold_obj', 'stand', 'sit_instr', 'ride_instr', 'walk', 'look_obj', 'hit_instr', 'hit_obj','eat_obj','eat_instr', 'jump_instr', 'lay_instr', 'talk_on_phone_instr', 'carry_obj',
                  'throw_obj', 'catch_obj', 'cut_instr','cut_obj' ,'run', 'work_on_computer_instr', 'ski_instr', 'surf_instr', 'skateboard_instr', 'smile', 'drink_instr',
                  'kick_obj', 'point_instr', 'read_obj', 'snowboard_instr']#67 89 1415
action_role = ['hit_instr', 'catch_obj', 'point_instr', 'cut_instr', 'work_on_computer_instr', 'lay_instr', 'look_obj', \
               'talk_on_phone_instr', 'snowboard_instr', 'smile', 'sit_instr', 'carry_obj', 'throw_obj', 'eat_obj',
               'walk', \
               'skateboard_instr', 'kick_obj', 'cut_obj', 'hold_obj', 'hit_obj', 'drink_instr', 'jump_instr',
               'ride_instr', \
               'stand', 'surf_instr', 'eat_instr', 'run', 'read_obj', 'ski_instr']

vcoco_single=[1,4,15,20]
vcoco_three=[6,7,14]
vcoco_obj=[0,5,11,12,13,22,24]
vcoco_instr=[2,3,8,9,10,16,17,18,19,21,23,25]
PREDICTED_RESULTS = '/home/priv-lab1/workspace/zxh/HOI/vcoco_det/results/map/multi_all/predicted/'
OUT_PUTDIR = './detection.pkl'

# Test_RCNN = pickle.load(
#     open('/home/priv-lab1/workspace/zxh/HOI/vcoco_det/data/' + 'Test_Faster_RCNN_R-50-PFN_2x_VCOCO.pkl', "rb"),
#     encoding='bytes')
# # for line in open('../DATA/'+ 'v-coco/data/splits/vcoco_test.ids', 'r'):
# #     image_id = int(line.rstrip())
# #     blobs = {}
# #     blobs['H_num'] = 1
# #     for Human_out in Test_RCNN[image_id]:
# #         dic = {}
# #         dic['image_id'] = image_id
# #         blobs['H_boxes'] = np.array([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]).reshape(1,5)
# #         dic['person_box'] = Human_out[2]
# #         print(dic['person_box'])
# #         for Object in Test_RCNN[image_id]:
# #             if (np.max(Object[5]) > object_thres) and not (np.all(Object[2] == Human_out[2])):  # This is a valid object
# #                 print()

agent_list = []
gt=False
for line in open('../DATA/' + 'v-coco/hoi-data/test/test_filename.txt', 'r'):
    image_id = int(line.rstrip().split('_')[2].split('.')[0])
    if os.path.exists(PREDICTED_RESULTS + os.path.splitext(line.rstrip())[0] + '.txt'):

        for res_line in open(PREDICTED_RESULTS + os.path.splitext(line.rstrip())[0] + '.txt', 'r'):
            dic = {}
            dic['image_id'] = image_id
            line_list = res_line.strip().split(' ')
            if gt:

                human_out = np.array([int(line_list[1]), int(line_list[2]), int(line_list[3]), int(line_list[4])])
                if int(line_list[5])==0 and int(line_list[6])==0 and  int(line_list[7])==0  and int(line_list[8])==0:
                    obj_out = np.full(4, np.nan).reshape(1, 4)
                else:
                    obj_out = np.array([int(line_list[5]), int(line_list[6]), int(line_list[7]), int(line_list[8])])


                if len(line_list)>9:
                    olabel= line_list[9]
                else:
                    olabel = ' '
            else:
                human_out = np.array([int(line_list[2]), int(line_list[3]), int(line_list[4]), int(line_list[5])])
                if int(line_list[6]) == 0 and int(line_list[7]) == 0 and int(line_list[8]) == 0 and int(
                        line_list[9]) == 0:
                    obj_out = np.full(4, np.nan).reshape(1, 4)
                else:
                    obj_out = np.array([int(line_list[6]), int(line_list[7]), int(line_list[8]), int(line_list[9])])
            dic['person_box'] = human_out
            action_id = int(line_list[0])
            action_score = float(1)#line_list[1]

            if action_id==12:
                agent_name =  'talk_on_phone_agent'
                dic[agent_name] = 1
            elif action_id==19:
                agent_name = 'work_on_computer_agent'
                dic[agent_name] = 1
            elif action_id==1:
                agent_name = 'stand_agent'
                dic[agent_name] = 1
            elif  action_id==4:
                agent_name = 'walk_agent'
                dic[agent_name] = 1
            elif  action_id==18:
                agent_name = 'run_agent'
                dic[agent_name] = 1
            elif  action_id==23:
                agent_name = 'smile_agent'
                dic[agent_name] = 1
            elif  action_id==7 or action_id==9 or action_id==17:
                pass
            else:
                agent_name = action_classes_29[action_id].split('_')[0] + '_agent'
                dic[agent_name] = 1
            for action in action_agent:
                if action not in dic:
                    dic[action] = 0.0
            dic[action_classes_29[action_id]] = np.append(obj_out, 1)
            for action in action_role:
                if action not in dic:
                    dic[action] = np.append(np.full(4, np.nan).reshape(1, 4), 0.0)
            detection.append(dic)




pickle.dump(detection, open(OUT_PUTDIR, "wb"))
