"""Add class id on the csv files

Author: @developmentseed

Run:
    python3 add_class_id.py \
           --csv=SL25_train_sliced_image_nbboxes.csv \
           --csv_output=SL25_train_sliced_image_nbboxes_class_id.csv
"""
import pandas as pd
import click


@click.command(short_help="create tfrecords for ml training")
@click.option('--csv', help="path to a csv that save", required=True, type=str)
@click.option('--csv_output', help="path to a csv class map", required=True, type=str)
def main(csv, csv_output):
    """Add label and group id in df and save as csv file
    Args:
        csv(string): csv file that contains the chip id and bboxes
        csv_output(string): csv output
    Returns:
        (None):
    """
    # Read config file
    config_df = pd.read_csv('training_data_stats/class_map.csv')
    class_map_dict = {}
    for index, row in config_df.iterrows():
        class_map_dict[row['label']] = {
            "category": row["category"],
            "label_id": row["label_id"],
            "master_label_id": row["master_label_id"],
            "group": row["group"],
            "group_id": row["group_id"],
            "master_group_id": row["master_group_id"]
        }

    # # Read the csv
    df = pd.read_csv(csv)
    df['label'] = df.label.apply(lambda x: x.lower())
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Drop large_birds and lion
    df = df[df['label'] != 'crane']
    df = df[df['label'] != 'ostrich']
    df = df[df['label'] != 'stork']
    df = df[df['label'] != 'lion']
    # complete with other atributes
    df['label'] = df.label.apply(lambda x: x.lower())
    df['category'] = df.label.apply(lambda x: class_map_dict[x]['category'])
    df['label_id'] = df.label.apply(lambda x: class_map_dict[x]['label_id'])
    df['master_label_id'] = df.label.apply(lambda x: class_map_dict[x]['master_label_id'])
    df['group'] = df.label.apply(lambda x: class_map_dict[x]['group'])
    df['group_id'] = df.label.apply(lambda x: class_map_dict[x]['group_id'])
    df['master_group_id'] = df.label.apply(lambda x: class_map_dict[x]['master_group_id'])
    # save csv
    df.to_csv(csv_output, index=False)

if __name__ == "__main__":
    main()
