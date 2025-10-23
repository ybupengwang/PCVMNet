class Config():
    # img_dir = 'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\TrainingData'
    # gt_dir = 'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\Training'
    # test_img_dir1 = 'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\Test1Data'
    # test_gt_dir1 = 'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\Test1'
    img_dir = r'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\antrainimg'
    gt_dir = r'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\antrainlabel'
    test_img_dir1 = r'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\antestimg'
    test_gt_dir1 = r'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\antestlabel'
    test_img_dir2 = 'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\Test2Data'
    test_gt_dir2 = 'D:\deeplearning\Anatomic-Landmark-Detection\process_data\cepha\Test2'
    GPU = 0
    optimizer = 'adam'
    base_number = 1
    # scal_h = 2400/512
    # scal_w = 1935/512
    resize_h = 512
    resize_w = 512
    sigma = 10
    point_num = 13
    num_epochs = 200
    lr = 1e-5
    # w = 1935
    # h = 2400
    trans = False
    struct_biaozhi = False
    cvm_biaozhi = True
    save_model_path = ''
    save_results_path = ''
    handimg_dir = r"D:\dataset\yachi\shou-biaozhu\set1\trainimage"
    handgt_dir = r"D:\dataset\yachi\shou-biaozhu\set1\traindata"
    testhandimg_dir = r"D:\dataset\yachi\shou-biaozhu\set1\testimage"
    testhandgt_dir = r"D:\dataset\yachi\shou-biaozhu\set1\testdata"
    handresize_h = 512
    handresize_w = 512



