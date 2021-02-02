"""
script to extract all labels from xml files from CVAT.
Note: these script will break if the column names is not matched with current ones;
      the categorize_impacts function will need to be updated with more labels come in.

Author: @developmentseed

Run:
    python3 xmls_to_df.py --xml_path=TA25 --csv_out=labeled_aiaia.csv
"""
import sys
import os
from os import path as op
import xml.etree.ElementTree as etree
import pandas as pd
import argparse


def parse_xml_attributes(xml):
    """parse xml and get all the attributes

    Args:
        xml: xml file contain bbox and label information

    Returns:
        attributes (list): a list of extracted attributes

    """
    attributes = []
    root= etree.parse(xml).getroot()
    image_entries = root.findall('image')

    for image in image_entries:
        width = int(image.get('width'))
        height = int(image.get('height'))
        for bb in image.findall('box'):
            image_id = image.get('name')
            label= bb.get('label')
            bbox = [float(bb.get(coord_key)) for coord_key in [ 'xtl', 'ytl', 'xbr', 'ybr']]
            attributes.append([image_id, label, bbox])
    return attributes

def dataframe_attributes(attributes, columns=None):
    """format attributes into pandas dataframe with column nane

    Args:
        attributes(list): a list of attributes
        columns: column names to be written in the dataframe
    """

    df = pd.DataFrame(attributes, columns=columns)
    return df

def df_all_attributes(xmls, columns = ['image_id', 'label','bbox']):
    """format all attribute to a collective padas dataframe

    Args:
        xmls(xml): xml files
        columns: pandas dataframe column names
    Returns:
        df_all: pandas dataframe saved attributes in designated columns
    """
    dfs = []
    for xml in xmls:
        attris = parse_xml_attributes(xml)
        df = dataframe_attributes(attris, columns=columns)
        dfs.append(df)
    df_all = pd.concat(dfs)
    return df_all

def categorize_impacts(given_val):
    """categorize impacts based on the given value
    TODOs: following lists need to refine by looking at df.groupby('label').agg('count')

    Args:
        given_val(str): given label;

    Returns:
        key(str): category of the label.
    """
    human_activities=['boma','building', 'charcoal mound', 'charcoal sack', 'human']
    livestock=['cow', 'donkey', 'shoats']
    wildlife=['eland', 'gazelle', 'giraffe', 'hartebeest', 'kudu',
              'oryx', 'buffalo', 'wildebeest', 'zebra', 'elephant', 'impala',
              'Ostrich', 'warthog', 'topi', 'waterbuck', 'Antelope', 'hippopotamus',
              'roan', 'sable', 'Lion','Crane', 'Stork']
    main_dic = dict(human_activities =human_activities,
                    livestock=livestock,
                    wildlife = wildlife)

    for key, value in main_dic.items():
        if given_val in value:
            return key

def parse_arg(args):
    desc = "xml_to_df"
    dhf = argparse.RawTextHelpFormatter
    parse0 = argparse.ArgumentParser(description= desc, formatter_class=dhf)
    parse0.add_argument('--xml_path', help="the directory that save all xml files")
    parse0.add_argument('--csv_out', help="the name to save a csv")
    return vars(parse0.parse_args(args))

def main(xml_path, csv_out):
    xmls=[op.join(xml_path, xml) for xml in os.listdir(xml_path) if xml.endswith('.xml')]
    df = df_all_attributes(xmls, columns = ['image_id', 'label','bbox'])
    df['category']=df['label'].apply(lambda x: categorize_impacts(x))
    df.to_csv(csv_out,index=False)

def cli():
    args = parse_arg(sys.argv[1:])
    main(**args)


if __name__ == "__main__":
    cli()
