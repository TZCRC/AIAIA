# Post inference processing scripts

## Drawing bbox on the images

This script is used to draw predicted bboxes and scores from an object detection model inference over the images. It intends to inspect the model prediction.


```bash
    python draw_bboxes_over_chip.py \
            --json_file=data/wildlife_results.json \
            --aws_bucket=aisurvey \
            --out_chip_dir=data/wildlife_inspect
```

Run all necesary steps 👇

```bash
    ./build_img_inspection.sh
```

## Building mbtiles for dashboard

- Merge json predictions and CSV geo-coordinates

```bash
    python3 match_pred_images.py \
        --csv_location_file=data/data/ai4earth_locations.csv \
        --json_prediction_file=data/human_activities_inference_results.json \
        --category=human_activities \
        --threshold=0.85 \
        --output_csv_file=data/human_activities_inference_results.csv \
        --output_geojson_file=data/human_activities_inference_results.geojson
```

Run all necesary steps 👇

```bash
    ./build_mbtiles.sh
```

## Building XML cvat format files

This script converts the json inference to CVAT xml , for later creating the task in CVAT and make human validation.


```bash
    python3 prediction2cvatxml.py \
        --csv_location_file=data/data/ai4earth_locations.csv \
        --json_prediction_file=data/inference_results.json \
        --category=human_activities \
        --threshold=0.85 \
        --output_xml_file=data/human_activities_inference_results.xml
```

Run all necesary steps 👇

```
./build_cvatxml.sh

```