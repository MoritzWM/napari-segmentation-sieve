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
        values, counts = np.unique_counts(data)
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


class MorphologyTools(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._label_layer_combo = create_widget(
            label="Layer", annotation="napari.layers.Labels"
        )
        self._btn_dilation = PushButton(text="Dilate")
        self._btn_erosion = PushButton(text="Erode")
        self._btn_open = PushButton(text="Open")
        self._btn_close = PushButton(text="Close")
        self._spin_radius = SpinBox(label="Radius", min=1, max=1e9)
        self._btn_dilation.clicked.connect(self._on_dilation_clicked)
        self._btn_erosion.clicked.connect(self._on_erosion_clicked)
        self._btn_open.clicked.connect(self._on_open_clicked)
        self._btn_close.clicked.connect(self._on_close_clicked)
        self._container_buttons = Container(
            layout="horizontal",
            widgets=[
                self._btn_dilation,
                self._btn_erosion,
                self._btn_open,
                self._btn_close,
            ],
        )

        self.extend(
            [
                self._label_layer_combo,
                self._spin_radius,
                self._container_buttons,
            ]
        )

    @with_layer_data(["_label_layer_combo"])
    def _on_dilation_clicked(self, data):
        radius = self._spin_radius.value
        return {"_label_layer_combo": isotropic_dilation(data, radius=radius)}

    @with_layer_data(["_label_layer_combo"])
    def _on_erosion_clicked(self, data):
        radius = self._spin_radius.value
        return {"_label_layer_combo": isotropic_erosion(data, radius=radius)}

    @with_layer_data(["_label_layer_combo"])
    def _on_open_clicked(self, data):
        radius = self._spin_radius.value
        return {"_label_layer_combo": isotropic_opening(data, radius=radius)}

    @with_layer_data(["_label_layer_combo"])
    def _on_close_clicked(self, data):
        radius = self._spin_radius.value
        return {"_label_layer_combo": isotropic_closing(data, radius=radius)}


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
