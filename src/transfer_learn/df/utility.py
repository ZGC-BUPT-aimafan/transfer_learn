import numpy as np

# Load data for non-defended dataset for CW setting


def split_dataset(fname, val_split=0.5, test_split=0.5):
    f = np.load(fname, allow_pickle=True)
    data = f["data"]
    labels = f["labels"]
    tmp_dic = {}
    for x, y in zip(data, labels):
        tmp_dic.setdefault(y, [])
        tmp_dic[y].append(x)
    for x in tmp_dic.keys():
        tmp_dic[x].reverse()
    x_train = []
    x_val = []
    x_test = []
    y_train = []
    y_val = []
    y_test = []
    for key in tmp_dic.keys():
        tmp_test_split = int(len(tmp_dic[key]) * (1 - test_split))
        tmp_val_split = int(len(tmp_dic[key]) * (1 - test_split - val_split))
        x_train.extend(tmp_dic[key][:tmp_val_split])
        for _ in range(len(tmp_dic[key][:tmp_val_split])):
            y_train.append(key)
        x_val.extend(tmp_dic[key][tmp_val_split:tmp_test_split])
        for _ in range(len(tmp_dic[key][tmp_val_split:tmp_test_split])):
            y_val.append(key)
        x_test.extend(tmp_dic[key][tmp_test_split:])
        for _ in range(len(tmp_dic[key][tmp_test_split:])):
            y_test.append(key)

    x_train = np.array(x_train)
    x_val = np.array(x_val)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    """
    num = x.shape[0]
    split = int(num * (1 - test_split))

    x_test = np.array(x[split:])
    y_test = np.array(y[split:])

    num = x.shape[0] - x_test.shape[0]
    split = int(num * (1 - val_split))

    x_val = np.array(x[split:num])
    y_val = np.array(y[split:num])

    x_train = np.array(x[:split])
    y_train = np.array(y[:split])
    """
    """
    f=np.load(fname,allow_pickle=True)
    tmp_x=np.array(f["data"])
    y=np.array(f["labels"])
    
    x=[]
    for i in range(len(tmp_x)):
        x.append(tmp_x[i].tolist()[:100])
    x=np.array(x)
    
    x=tmp_x
    print(x.shape)
    num = x.shape[0]
    split = int(num * (1 - test_split))

    x_test = np.array(x[split:])
    y_test = np.array(y[split:])
    
    #num = x.shape[0] - x_test.shape[0]
    split2 = int(num * (1 - val_split-test_split))

    x_val = np.array(x[split2:split])
    y_val = np.array(y[split2:split])

    x_train = np.array(x[:split2])
    y_train = np.array(y[:split2])

    val_dic={}
    test_dic={}
    train_dic={}
    tmp_train=y_train.tolist()
    tmp_val=y_val.tolist()
    tmp_test=y_test.tolist()
    
    for i in range(0,2):
        num=i
        train_dic[num]=tmp_train.count(num)
    for i in range(0,2):
        num=i
        val_dic[num]=tmp_val.count(num)
    for i in range(0,2):
        num=i
        test_dic[num]=tmp_test.count(num)
    print(train_dic)
    print(val_dic)
    print(test_dic)
    # print(y_val)
    # print(y_test)
    """
    print("Data dimensions:")
    print("X: Training data's shape : ", x_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", x_val.shape)
    print("y: Validation data's shape : ", y_val.shape)
    print("X: Testing data's shape : ", x_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test
