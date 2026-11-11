import numpy as np
from napari_segmentation_sieve._widget import (
    ExampleQWidget,
    ImageThreshold,
    threshold_autogenerate_widget,
    threshold_magic_widget,
)


def test_threshold_autogenerate_widget():
    # because our "widget" is a pure function, we can call it and
    # test it independently of napari
    im_data = np.random.random((100, 100))
    thresholded = threshold_autogenerate_widget(im_data, 0.5)
    assert thresholded.shape == im_data.shape
    # etc.


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# you don't need to import it, as long as napari is installed
# in your testing environment
def test_threshold_magic_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))

    # our widget will be a MagicFactory or FunctionGui instance
    my_widget = threshold_magic_widget()

    # if we "call" this object, it'll execute our function
    thresholded = my_widget(viewer.layers[0], 0.5)
    assert thresholded.shape == layer.data.shape
    # etc.


def test_image_threshold_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))
    my_widget = ImageThreshold(viewer)

    # because we saved our widgets as attributes of the container
    # we can set their values without having to "interact" with the viewer
    my_widget._image_layer_combo.value = layer
    my_widget._threshold_slider.value = 0.5

    # this allows us to run our functions directly and ensure
    # correct results
    my_widget._threshold_im()
    assert len(viewer.layers) == 2


# capsys is a pytest fixture that captures stdout and stderr output streams
def test_example_q_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    my_widget = ExampleQWidget(viewer)

    # call our widget method
    my_widget._on_click()

    # read captured output and check that it's as we expected
    captured = capsys.readouterr()
    assert captured.out == "napari has 1 layers\n"


def test_split_labels_one_layer_per_value():
    from napari_segmentation_sieve._widget import split_labels

    data = np.zeros((10, 10), dtype=np.uint8)
    data[1:3, 1:3] = 1
    data[6:8, 5:9] = 4
    layers = split_labels(data, "seg")
    assert [kw["name"] for _, kw, _ in layers] == ["seg_1", "seg_4"]
    # each layer keeps the original shape and contains only one object, 1
    for arr, _, _ in layers:
        assert arr.shape == data.shape
        assert set(np.unique(arr)) == {0, 1}
    assert np.array_equal(layers[0][0], (data == 1).astype(np.uint8))
    assert np.array_equal(layers[1][0], (data == 4).astype(np.uint8))
    # no translation is added
    assert all("translate" not in kw for _, kw, _ in layers)


def test_split_then_merge_round_trip_renumbers_objects():
    from napari_segmentation_sieve._widget import merge_labels, split_labels

    data = np.zeros((20, 30, 40), dtype=np.uint16)
    data[2:6, 3:8, 4:9] = 1
    data[10:15, 20:25, 30:38] = 5
    parts = split_labels(data, "seg")
    merged = merge_labels([arr for arr, _, _ in parts])
    # same objects, renumbered 1..N in the order of the split layers
    assert np.array_equal(merged > 0, data > 0)
    assert set(np.unique(merged)) == {0, 1, 2}
    assert np.array_equal(merged == 1, data == 1)
    assert np.array_equal(merged == 2, data == 5)


def test_merge_labels_renumbers_multi_label_layers():
    from napari_segmentation_sieve._widget import merge_labels

    a = np.array([[3, 0, 0], [0, 7, 0]])  # labels 3 and 7 -> 1 and 2
    b = np.array([[0, 4, 0], [0, 0, 9]])  # labels 4 and 9 -> 3 and 4
    merged = merge_labels([a, b])
    assert np.array_equal(merged, np.array([[1, 3, 0], [0, 2, 4]]))


def test_merge_labels_rejects_different_shapes():
    import pytest
    from napari_segmentation_sieve._widget import merge_labels

    with pytest.raises(ValueError):
        merge_labels([np.zeros((2, 2)), np.zeros((3, 3))])


def test_split_labels_keeps_colors():
    from napari.utils.colormaps import DirectLabelColormap
    from napari_segmentation_sieve._widget import split_labels

    data = np.zeros((4, 4), dtype=np.uint8)
    data[0, 0] = 2
    data[3, 3] = 9
    colors = {2: (1.0, 0.0, 0.0, 1.0), 9: (0.0, 0.0, 1.0, 1.0)}
    layers = split_labels(data, "seg", colors=colors)
    cm0 = layers[0][1]["colormap"]
    cm1 = layers[1][1]["colormap"]
    assert isinstance(cm0, DirectLabelColormap)
    # each split layer's object (label 1) carries its original colour
    assert np.allclose(cm0.color_dict[1], colors[2])
    assert np.allclose(cm1.color_dict[1], colors[9])
