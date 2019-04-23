import pickle
Test_RCNN      = pickle.load( open( cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_VCOCO.pkl', "rb" ) )
prior_mask     = pickle.load( open( cfg.DATA_DIR + '/' + 'prior_mask.pkl', "rb" ) )
Action_dic     = json.load(   open( cfg.DATA_DIR + '/' + 'action_index.json'))
Action_dic_inv = {y:x for x,y in Action_dic.iteritems()}