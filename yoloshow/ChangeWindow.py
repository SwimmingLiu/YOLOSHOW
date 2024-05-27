from utils import glo


def yoloshowvsSHOW():
    yoloshow_glo = glo.get_value('yoloshow')
    yoloshowvs_glo = glo.get_value('yoloshowvs')
    glo.set_value('yoloname1', "yolov5 yolov7 yolov8 yolov9 yolov10 yolov5-seg yolov8-seg rtdetr yolov8-pose yolov8-obb")
    glo.set_value('yoloname2', "yolov5 yolov7 yolov8 yolov9 yolov10 yolov5-seg yolov8-seg rtdetr yolov8-pose yolov8-obb")
    yoloshowvs_glo.reloadModel()
    yoloshowvs_glo.show()
    yoloshow_glo.animation_window = None
    yoloshow_glo.closed.disconnect()

def yoloshowSHOW():
    yoloshow_glo = glo.get_value('yoloshow')
    yoloshowvs_glo = glo.get_value('yoloshowvs')
    glo.set_value('yoloname', "yolov5 yolov7 yolov8 yolov9 yolov10 yolov5-seg yolov8-seg rtdetr yolov8-pose yolov8-obb")
    yoloshow_glo.reloadModel()
    yoloshow_glo.show()
    yoloshowvs_glo.animation_window = None
    yoloshowvs_glo.closed.disconnect()
def yoloshow2vs():
    yoloshow_glo = glo.get_value('yoloshow')
    yoloshow_glo.closed.connect(yoloshowvsSHOW)
    yoloshow_glo.close()

def vs2yoloshow():
    yoloshowvs_glo = glo.get_value('yoloshowvs')
    yoloshowvs_glo.closed.connect(yoloshowSHOW)
    yoloshowvs_glo.close()