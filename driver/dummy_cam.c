#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/slab.h>
#include <media/v4l2-subdev.h>
#include <media/media-entity.h>

struct dummy_cam {
    struct v4l2_subdev sd;
    struct media_pad pad;
    struct v4l2_mbus_framefmt fmt;
};

static int dummy_get_fmt(struct v4l2_subdev *sd, struct v4l2_subdev_state *state, struct v4l2_subdev_format *format) {
    struct dummy_cam *dev = container_of(sd, struct dummy_cam, sd);
    format->format = dev->fmt;
    return 0;
}

static int dummy_set_fmt(struct v4l2_subdev *sd, struct v4l2_subdev_state *state, struct v4l2_subdev_format *format) {
    struct dummy_cam *dev = container_of(sd, struct dummy_cam, sd);
    dev->fmt = format->format;
    return 0;
}

static int dummy_s_stream(struct v4l2_subdev *sd, int enable) {
    return 0; 
}

static const struct v4l2_subdev_pad_ops dummy_pad_ops = {
    .get_fmt = dummy_get_fmt,
    .set_fmt = dummy_set_fmt,
};

static const struct v4l2_subdev_video_ops dummy_video_ops = {
    .s_stream = dummy_s_stream,
};

static const struct v4l2_subdev_ops dummy_ops = {
    .pad = &dummy_pad_ops,
    .video = &dummy_video_ops,
};

static int dummy_probe(struct platform_device *pdev) {
    struct dummy_cam *dev = devm_kzalloc(&pdev->dev, sizeof(*dev), GFP_KERNEL);
    if (!dev) return -ENOMEM;

    v4l2_subdev_init(&dev->sd, &dummy_ops);
    dev->sd.dev = &pdev->dev;
    dev->sd.owner = THIS_MODULE;
    
    dev->sd.flags |= V4L2_SUBDEV_FL_HAS_DEVNODE; 
    snprintf(dev->sd.name, sizeof(dev->sd.name), "dummy-csi2");
    
    // NATIVE FIX: Set format to 8-bit to align with RAW8 (Data Type 0x2A)
    dev->fmt.code = MEDIA_BUS_FMT_Y8_1X8;
    dev->fmt.width = 100;
    dev->fmt.height = 38;
    
    dev->fmt.field = V4L2_FIELD_NONE;
    dev->fmt.colorspace = V4L2_COLORSPACE_RAW;

    dev->pad.flags = MEDIA_PAD_FL_SOURCE;
    dev->sd.entity.function = MEDIA_ENT_F_CAM_SENSOR;
    
    if (media_entity_pads_init(&dev->sd.entity, 1, &dev->pad)) return -ENODEV;
    platform_set_drvdata(pdev, &dev->sd);

    return v4l2_async_register_subdev(&dev->sd);
}

static const struct of_device_id dummy_dt_ids[] = {
    { .compatible = "raspberrypi,dummy-csi2-sensor" },
    { }
};
MODULE_DEVICE_TABLE(of, dummy_dt_ids);

static struct platform_driver dummy_driver = {
    .driver = {
        .name = "dummy_cam",
        .of_match_table = dummy_dt_ids,
    },
    .probe = dummy_probe,
};
module_platform_driver(dummy_driver);
MODULE_LICENSE("GPL");