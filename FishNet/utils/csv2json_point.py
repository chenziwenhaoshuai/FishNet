import json
file = "../json_output/point.csv"
jsonpath = "../dataset_test/annotations/point_label.json" # 标签
# label_path = "./labels/"
# labelsss = []
output = {}
name = set()
file_name = []
with open(jsonpath,'r') as ojson:
    load_dict = json.load(ojson)
    images = load_dict['images']
with open(file, 'r') as csv:
    csv = csv.read().split('\n')
    for line in csv:
        if len(line)>3:
            line = line.split(',')
            cls = line[0]
            file_name.append(line[1])
            conf = line[2]
            x = int(line[3])
            y = int(line[4])
            w = int(line[5])-int(line[3])
            h = int(line[6])-int(line[4])
    # print(file_name)
    name = list(set(file_name))
    # images = []
    # for index,imgname in enumerate(name):
    #     bianhao = {"file_name":imgname,"id":index}
    #     # bianhao = json.dumps(bianhao)
    #     images.append(bianhao)
    #     print(bianhao)
    annotations = []
    for line in csv:
        if len(line)>3:
            line = line.split(',')

            file_name = line[1]
            for lines in images:
                if file_name == lines["file_name"]:
                    id = lines["id"]
                    conf = float(line[2])
                    cls = int(line[0])
                    x = int(line[3])
                    y = int(line[4])
                    w = int(line[5])-int(line[3])
                    h = int(line[6])-int(line[4])
                    points =list(map(int,line[3:-1]))
                    kuang = {"image_id":id,"category_id":cls,"keypoints":points,"score":conf}
                    # # kuang = json.dumps(kuang)
                    # kuang = kuang.replace('\'','\"')
                    # with open('../dataset_test/annotations/x6-res.json', 'a') as json:
                    #     json.write(str(kuang))
                    annotations.append(kuang)
                    print(kuang)
    # output = {"images":images,"annotations":annotations}
    output = annotations
    output = json.dumps(output)
    output = output.replace('\'','\"')
    with open('../json_output/point.json', 'a') as json:
        json.write(str(output))
