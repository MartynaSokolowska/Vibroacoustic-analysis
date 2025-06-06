from dataclasses import dataclass
from typing import List, Tuple, Union


@dataclass
class MovementSegment:
    name: str
    length_cm: Union[float, Tuple[float, float]]
    going_down_time: Union[float, Tuple[float, float]]
    going_up_time: Union[float, Tuple[float, float]]

    def __str__(self):
        length = (
            f"{self.length_cm} cm"
            if isinstance(self.length_cm, float)
            else f"{self.length_cm[0]} / {self.length_cm[1]} cm"
        )
        return (
            f"{self.name}: Length = {length}\n"
            f"  Going Down Time: {self.going_down_time}\n"
            f"  Going Up Time: {self.going_up_time}\n"
        )


movement_segmentations_raw = {
    "slow": {
        "zucchini_top": [
            MovementSegment("Above gelatine", 6.5, 0.00, (12.13, 12.23)),
            MovementSegment("Gelatine", 6.0, (0.97, 1.03), (11.53, 11.60)),
            MovementSegment("Zuchini Top", 5.4, (1.43, 1.57), (11.10, 11.17)),
            MovementSegment("Zuchini Bottom / Chicken Top", (3.0, 2.9), (3.07, 3.20), (9.40, 9.53)),
            MovementSegment("Chicken Bottom", (1.1, 0.8), (4.43, 4.67), (8.00, 8.13)),
            MovementSegment("Gelatine (inside)", -0.5, (5.77, 5.90), (6.87, 6.93)),
        ],
        "zucchini_bottom": [
            MovementSegment("Above gelatine", 6.5, 0.00, (11.53, 11.60)),
            MovementSegment("Gelatine", 6.0, (0.97, 1.03), (10.90, 10.97)),
            MovementSegment("Chicken Top", 5.4, (1.40, 1.47), (10.47, 10.53)),
            MovementSegment("Chicken Bottom / Zucchini Top", 3.4, (2.80, 2.87), (9.10, 9.17)),
            MovementSegment("Zucchini Bottom", 1.4, (4.17, 4.23), (7.70, 7.77)),
            MovementSegment("Gelatine (inside)", (0.3, 0.4), (5.47, 5.53), (6.53, 6.60)),
        ]
    },
    "medium": {
        "zucchini_top": [
            MovementSegment("Above gelatine", 6.5, 0.00, (9.17, 9.30)),
            MovementSegment("Gelatine", 6.0, (0.83, 0.90), (8.70, 8.77)),
            MovementSegment("Zuchini Top", 5.4, (1.17, 1.20), (8.37, 8.43)),
            MovementSegment("Zuchini Bottom / Chicken Top", (3.0, 2.9), (2.27, 2.33), (7.23, 7.30)),
            MovementSegment("Chicken Bottom", (1.1, 0.8), (3.17, 3.27), (6.23, 6.40)),
            MovementSegment("Gelatine (inside)", -0.5, (4.17, 4.33), (5.30, 5.63)),
        ],
        "zucchini_bottom": [
            MovementSegment("Above gelatine", 6.5, 0.00, 8.87),
            MovementSegment("Gelatine", 6.0, (0.83, 0.87), (8.27, 8.33)),
            MovementSegment("Chicken Top", 5.4, (1.13, 1.17), (7.97, 8.00)),
            MovementSegment("Chicken Bottom / Zucchini Top", 3.4, (2.10, 2.13), (7.00, 7.03)),
            MovementSegment("Zucchini Bottom", 1.4, (3.00, 3.03), (6.07, 6.13)),
            MovementSegment("Gelatine (inside)", (0.3, 0.4), (4.03, 4.10), (5.07, 5.40)),
        ]
    },
    "fast": {
        "zucchini_top": [
            MovementSegment("Above gelatine", 6.5, 0.00, (6.87, 6.93)),
            MovementSegment("Gelatine", 6.0, (0.67, 0.70), (6.50, 6.63)),
            MovementSegment("Zuchini Top", 5.4, (0.94, 0.97), (6.27, 6.30)),
            MovementSegment("Zuchini Bottom / Chicken Top", (3.0, 2.9), (1.60, 1.67), (5.57, 5.63)),
            MovementSegment("Chicken Bottom", (1.1, 0.8), (2.17, 2.30), (4.87, 5.03)),
            MovementSegment("Gelatine (inside)", -0.5, (3.03, 3.10), (4.10, 4.43)),
        ],
        "zucchini_bottom": [
            MovementSegment("Above gelatine", 6.5, 0.00, (6.67, 6.73)),
            MovementSegment("Gelatine", 6.0, (0.67, 0.70), (6.27, 6.30)),
            MovementSegment("Chicken Top", 5.4, (0.93, 0.97), (6.03, 6.07)),
            MovementSegment("Chicken Bottom / Zucchini Top", 3.4, (1.50, 1.53), (5.47, 5.50)),
            MovementSegment("Zucchini Bottom", 1.4, (2.07, 2.10), (4.87, 4.90)),
            MovementSegment("Gelatine (inside)", (0.3, 0.4), (2.73, 2.80), (4.03, 4.30)),
        ]
    }
}


class MaterialSegments:
    def __init__(self, material_name: str):
        self.material_name = material_name
        self.segments: List[Tuple[float, float]] = []

    def add_segment(self, start: float, end: float):
        self.segments.append((start, end))

    def __repr__(self):
        return f"<{self.material_name}: {self.segments}>"


def get_material_segments(movement_type: str, layout_type: str):
    chicken = MaterialSegments("chicken")
    gelatine = MaterialSegments("gelatine")
    zucchini = MaterialSegments("zucchini")

    movement_segmentation = movement_segmentations_raw[movement_type][layout_type]

    gelatine.add_segment(movement_segmentation[1].going_down_time[1], movement_segmentation[2].going_down_time[0])

    if layout_type == "zucchini_top":
        zucchini.add_segment(movement_segmentation[2].going_down_time[1], movement_segmentation[3].going_down_time[0])
        chicken.add_segment(movement_segmentation[3].going_down_time[1], movement_segmentation[4].going_down_time[0])

    if layout_type == "zucchini_bottom":
        chicken.add_segment(movement_segmentation[2].going_down_time[1], movement_segmentation[3].going_down_time[0])
        zucchini.add_segment(movement_segmentation[3].going_down_time[1], movement_segmentation[4].going_down_time[0])

    gelatine.add_segment(movement_segmentation[4].going_down_time[1], movement_segmentation[5].going_down_time[0])

    return [chicken, gelatine, zucchini]


segmentations = {
    "slow": {
        "zucchini_top": get_material_segments("slow", "zucchini_top"),
        "zucchini_bottom": get_material_segments("slow", "zucchini_bottom"),
    },
    "medium": {
        "zucchini_top": get_material_segments("medium", "zucchini_top"),
        "zucchini_bottom": get_material_segments("medium", "zucchini_bottom"),
    },
    "fast":{
        "zucchini_top": get_material_segments("fast", "zucchini_top"),
        "zucchini_bottom": get_material_segments("fast", "zucchini_bottom"),
    }
}