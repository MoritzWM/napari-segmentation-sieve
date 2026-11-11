from typing import TYPE_CHECKING

import numpy as np
from magicgui import magic_factory
from magicgui.widgets import (
    Container,
    PushButton,
    SpinBox,
    Table,
    create_widget,
)
from scipy import ndimage
from scipy.ndimage import binary_erosion
from skimage import img_as_float
from skimage.morphology import (
    isotropic_closing,
    isotropic_dilation,
    isotropic_erosion,
    isotropic_opening,
    remove_small_objects,
)
from skimage.segmentation import watershed

if TYPE_CHECKING:
    import napari


def with_layer_data(layer_combos, callback=None):
    def decorator(method):
        def wrapper(self):
            layers = {}
            data_list = []

            for combo_name in layer_combos:
                layer = getattr(self, combo_name).value
                if layer is None:
                    return
                data = layer.data
                if data is None:
                    return
                layers[combo_name] = layer
                data_list.append(data)

            # Call method with unpacked data
            if len(data_list) == 1:
                result = method(self, data_list[0])
            else:
                result = method(self, *data_list)

            # Update layers from result dict
            if isinstance(result, dict):
                for combo_name, data in result.items():
                    if combo_name in layers:
                        layers[combo_name].data = data

            # Invoke callback if provided
            if callback:
                if isinstance(callback, str):
                    getattr(self, callback)()
                else:
                    callback(self)

        return wrapper

    return decorator


class SegmentationSieve(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._label_layer_combo = create_widget(
            label="Layer", annotation="napari.layers.Labels"
        )
        self._label_layer_combo.changed.connect(
            self._update_table_voxel_counts
        )
        # Renumbering
        self._btn_renumber = PushButton(
            text="Renumber objects",
        )
        self._btn_renumber_to_one = PushButton(
            text="Renumber objects to 1",
        )
        self._container_renumber = Container(
            label="Renumbering",
            layout="horizontal",
            widgets=[
                self._btn_renumber,
                self._btn_renumber_to_one,
            ],
        )
        self._btn_renumber.clicked.connect(self._on_renumber_clicked)
        self._btn_renumber_to_one.clicked.connect(
            self._on_renumber_to_one_clicked
        )
        # Small objects removal
        self._spin_small_object_thresh = SpinBox(min=1, max=1e9)
        self._btn_small_objects = PushButton(text="Remove")
        self._btn_small_objects.clicked.connect(
            self._on_remove_small_objects_clicked
        )
        self._container_small_objects = Container(
            label="Small object removal",
            layout="horizontal",
            widgets=[
                self._spin_small_object_thresh,
                self._btn_small_objects,
            ],
        )
        # Voxel counts
        self._table_voxel_counts = Table()
        self.extend(
            [
                self._label_layer_combo,
                self._container_renumber,
                self._container_small_objects,
                self._table_voxel_counts,
            ]
        )

    def _update_table_voxel_counts(self):
        layer = self._label_layer_combo.value
        if layer is None:
            return
        data = layer.data
        if data is None:
            return
        values, counts = np.unique(data, return_counts=True)
        self._table_voxel_counts.value = {
            "Label": values,
            "Voxels": counts,
        }

    @with_layer_data(
        ["_label_layer_combo"], callback="_update_table_voxel_counts"
    )
    def _on_renumber_clicked(self, data):
        new_data, _ = ndimage.label(data, structure=np.ones((3, 3, 3)))
        return {"_label_layer_combo": new_data}

    @with_layer_data(
        ["_label_layer_combo"], callback="_update_table_voxel_counts"
    )
    def _on_renumber_to_one_clicked(self, data):
        return {"_label_layer_combo": data > 0}

    @with_layer_data(
        ["_label_layer_combo"], callback="_update_table_voxel_counts"
    )
    def _on_remove_small_objects_clicked(self, data):
        thresh = self._spin_small_object_thresh.value
        return {
            "_label_layer_combo": remove_small_objects(data, max_size=thresh)
        }


@magic_factory(
    operation={"choices": ["dilate", "erode", "open", "close", "outer_hull"]},
    radius={"min": 1, "max": 1e9},
)
def morphology_widget(
    layer: "napari.layers.Labels",
    operation: str,
    radius: int = 1,
) -> "napari.types.LabelsData":
    match operation:
        case "dilate":
            return isotropic_dilation(layer.data, radius=radius)
        case "erode":
            return isotropic_erosion(layer.data, radius=radius)
        case "open":
            return isotropic_opening(layer.data, radius=radius)
        case "close":
            return isotropic_closing(layer.data, radius=radius)
        case "outer_hull":
            # Create a copy with only the outer surface voxels
            surface_voxels = layer.data > 0
            interior = binary_erosion(
                surface_voxels, structure=np.ones((3, 3, 3))
            )
            surface_voxels[interior] = 0
            return surface_voxels
        case _:
            raise ValueError(f"Invalid operation: {operation}")


@magic_factory(
    operation={"choices": ["add", "subtract", "multiply", "divide"]}
)
def arithmetics(
    layer_1: "napari.layers.Labels",
    layer_2: "napari.layers.Labels",
    operation: str,
) -> "napari.types.LabelsData":
    if layer_1.data.shape != layer_2.data.shape:
        raise ValueError("Can only combine two layers with the same shape")
    match operation:
        case "add":
            return np.add(layer_1.data, layer_2.data)
        case "subtract":
            return np.subtract(layer_1.data, layer_2.data)
        case "multiply":
            return np.multiply(layer_1.data, layer_2.data)
        case "divide":
            return np.divide(layer_1.data, layer_2.data)
        case _:
            raise ValueError(f"Invalid operation: {operation}")


def _on_threshold_widget_init(widget):
    @widget.img_layer.changed.connect
    def _on_layer_changed(image_layer):
        if image_layer is None:
            return
        widget.threshold.min = image_layer.data.min()
        widget.threshold.max = image_layer.data.max()

    _on_layer_changed(widget.img_layer.value)
    if widget.img_layer.value is not None:
        widget()


@magic_factory(
    widget_init=_on_threshold_widget_init,
    threshold={"widget_type": "FloatSlider"},
    auto_call=True,
)
def threshold_widget(
    img_layer: "napari.layers.Image", threshold: "float"
) -> "napari.types.LabelsData":
    return img_as_float(img_layer.data) > threshold


@magic_factory()
def watershed_widget(
    image_layer: "napari.layers.Image",
    mask_layer: "napari.layers.Labels | None",
    seed_layer: "napari.layers.Labels",
) -> "napari.types.LabelsData":
    return watershed(
        -image_layer.data,
        seed_layer.data,
        mask=mask_layer.data if mask_layer else None,
    )


@magic_factory(
    filter_name={"choices": ["gaussian"]},
    sigma={"min": 0, "max": 1e6},
)
def filters_widget(
    layer: "napari.layers.Image",
    filter_name: str,
    sigma: float,
) -> "napari.types.ImageData":
    from skimage import filters

    match filter_name:
        case "gaussian":
            return filters.gaussian(layer.data, sigma=sigma)


def _count_objects(layer):
    if layer is None:
        return 0
    values = np.unique(layer.data)
    return int(np.count_nonzero(values))


def _on_split_labels_widget_init(widget):
    def _update_call_button(layer=None):
        n = _count_objects(widget.layer.value)
        noun = "layer" if n == 1 else "layers"
        widget.call_button.text = f"Split into {n} {noun}"

    widget.layer.changed.connect(_update_call_button)
    _update_call_button()


@magic_factory(
    widget_init=_on_split_labels_widget_init,
    layer={"label": "Labels layer"},
    name_prefix={
        "label": "Name prefix",
        "tooltip": "Defaults to the layer name",
    },
)
def split_labels_widget(
    layer: "napari.layers.Labels",
    name_prefix: str = "",
) -> "list[napari.types.LayerDataTuple]":
    """Split a labels layer into one labels layer per object.

    Each new layer has the same shape as the original and contains a single
    object, labelled 1.
    """
    values = np.unique(layer.data)
    colors = {int(v): layer.get_color(v) for v in values if v != 0}
    return split_labels(
        layer.data, name_prefix or layer.name, layer.scale, colors
    )


def split_labels(data, name_prefix, scale=None, colors=None):
    """Return napari LayerDataTuples, one full-size layer per label value.

    ``colors`` maps original label values to RGBA. Each new layer's object
    (label 1) is given the colour its original label had.
    """
    from napari.utils.colormaps import DirectLabelColormap

    data = np.asarray(data)
    layers = []
    for value in np.unique(data):
        if value == 0:
            continue
        obj = np.zeros(data.shape, dtype=data.dtype)
        obj[data == value] = 1
        kwargs = {"name": f"{name_prefix}_{value}"}
        if scale is not None:
            kwargs["scale"] = scale
        color = (colors or {}).get(int(value))
        if color is not None:
            kwargs["colormap"] = DirectLabelColormap(
                color_dict={1: color, None: (0, 0, 0, 0)}
            )
        layers.append((obj, kwargs, "labels"))
    return layers


@magic_factory(call_button="Merge selected layers")
def merge_labels_widget(
    viewer: "napari.viewer.Viewer",
) -> "napari.types.LayerDataTuple":
    """Merge all selected labels layers into one new labels layer.

    All selected layers must have the same shape. Labels are always
    renumbered: each layer's labels get the next free label numbers, in
    their original order. Where layers overlap, the later layer in the
    selection wins.
    """
    from napari.layers import Labels

    layers = [
        layer for layer in viewer.layers.selection if isinstance(layer, Labels)
    ]
    if not layers:
        raise ValueError("Select one or more labels layers to merge")
    data = merge_labels([layer.data for layer in layers])
    return (
        data,
        {"name": "merged labels", "scale": layers[0].scale},
        "labels",
    )


def merge_labels(arrays):
    """Combine same-shaped label arrays, renumbering all labels.

    Each array's labels are mapped to consecutive numbers after the labels
    already used by earlier arrays.
    """
    if not arrays:
        raise ValueError("No labels layers given")
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise ValueError("All selected labels layers must have the same shape")
    max_label = sum(int(np.unique(a[a > 0]).size) for a in arrays)
    dtype = np.result_type(np.min_scalar_type(max_label), *arrays)
    out = np.zeros(shape, dtype=dtype)
    next_free = 0
    for data in arrays:
        values = np.unique(data[data > 0])
        if values.size == 0:
            continue
        mask = data > 0
        ranks = np.searchsorted(values, data[mask])
        out[mask] = next_free + 1 + ranks
        next_free += values.size
    return out
