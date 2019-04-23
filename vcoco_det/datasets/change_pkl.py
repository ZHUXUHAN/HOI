import pickle
detections_file ='/home/priv-lab1/workspace/zxh/HOI/vcoco_det/detections.pkl'
detection_file ='/home/priv-lab1/workspace/zxh/HOI/vcoco_det/detection.pkl'
with open(detections_file, 'rb') as f:
    detss = pickle.load(f,encoding='bytes')
with open(detection_file, 'rb') as f:
    dets = pickle.load(f,encoding='bytes')
# for det in dets:
#     det.update({'eat_agent': det.pop(b"eat_agent")})
#     det.update({'hit_agent': det.pop(b"hit_agent")})
#
#     det.update({'cut_agent': det.pop(b"cut_agent")})
#     det.update({'person_box': det.pop(b"person_box")})
#     det.update({'image_id': det.pop(b"image_id")})
#     det.update({'work_on_computer_agent': det.pop(b"work_on_computer_agent")})
#     det.update({'talk_on_phone_agent': det.pop(b"talk_on_phone_agent")})
#     print(det.keys())
# with open(detections_file, 'wb') as f:
#     pickle.dump(dets,f)

# img_ids=[]
for det in dets[:10]:#det是个字典
    print(det)

# for det in detss:
#     print(det['image_id'])
#
# print(len(detss ),len(dets))

# for i,det1 in enumerate(detss[0]):
# for ii,det2 in enumerate(detss):
#     if dets['image_id']==det2['image_id']==22:
#         # print(dets[2]['image_id'],dets[0])
#         # print(dets[2]['image_id'],det2)
#         for k,v in dets.items():
#             if k =='person_box':
#                 print(k,v)
#                 print(k,detss[ii][k])
        # print(detss[0].keys())
        # print("      ")
        # print(dets[ii].keys())
        # print("      ")


