#############################################
# List of all wsi_slides present in label_data but not in zoorvival wsi embedded. List of all patiets present in label_data that are missing from zoorvival wsi embeddings.
# the lists were obtained using the notebook 'check_missing_wsi.ipynb'.

# The code removes these patients from the datasets_csv files.
# present the code to remove the lists from each dataset file that contains patients (label_data, omics_data, clinical_data), each dataset separate
# and the code to do everything in one go (do not use, the code was not updated). 
# The functions were used directly in the code during the run, currently commented out because the (new) datasets_csv files are already cleaned --> see notebook 'clean_datasets.ipynb'
#############################################


import torch

#----> general imports
import pandas as pd
import numpy as np
import pdb
import os

# some slide_ids present in the datasets_csv from the original repository are not present in the zoorvival dataset.
# it was decided to remove them from the list of slide_ids to avoid errors.

# global list of wsi_ids to delete:
list_to_delete = ['TCGA-D8-A27R-01Z-00-DX1.F6E2FD1C-0666-4788-8D95-A76D15907270.svs', 
'TCGA-OK-A5Q2-01Z-00-DX1.0D169898-37C6-44CA-AC87-27887123AA6F.svs', 'TCGA-D8-A1XR-01Z-00-DX2.A103FB8B-4397-4DD4-8587-90A736407484.svs', 
'TCGA-A7-A13D-01Z-00-DX1.D206783C-FA6A-4B6A-B3AA-4132A2C9626B.svs', 'TCGA-S3-AA15-01Z-00-DX2.915A4F90-25CB-4535-99C7-D0D0CFC90412.svs', 
'TCGA-D8-A1JI-01Z-00-DX1.9BDB647F-EEAB-4235-BE44-A3815A48CCE0.svs', 'TCGA-EW-A423-01Z-00-DX2.5EF1CF39-600A-4ED3-A5A2-AF4435A5F8B5.svs', 
'TCGA-D8-A1X5-01Z-00-DX1.81B10B43-0D99-44D0-A245-D652041B8FEE.svs', 'TCGA-D8-A1XY-01Z-00-DX2.33D96E5C-5291-4864-B282-8BACA2043586.svs', 
'TCGA-D8-A1XL-01Z-00-DX1.FDF07020-8F40-4C00-9023-E5F40E0D8A7C.svs', 'TCGA-PL-A8LY-01A-01-DX1.32047D5E-8A42-480A-B2B8-A56B47B949FD.svs', 
'TCGA-D8-A1Y1-01Z-00-DX2.B58DC955-F864-4E78-8B1A-8156E2F7D554.svs', 'TCGA-D8-A1XW-01Z-00-DX2.9849E503-BE3E-417C-ABE8-93A39583DDE0.svs', 
'TCGA-A7-A0CD-01Z-00-DX1.F045B9C8-049C-41BF-8432-EF89F236D34D.svs', 'TCGA-D8-A3Z6-01Z-00-DX1.4076A770-5901-4325-85C3-AF7B192272F5.svs', 
'TCGA-S3-AA12-01Z-00-DX1.2A991B9F-E7E6-410B-B0BF-635E3CC40C7E.svs', 'TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.svs', 
'TCGA-OK-A5Q2-01Z-00-DX4.83B45D6C-E350-4436-812F-4155D9F7D331.svs', 'TCGA-D8-A27N-01Z-00-DX2.EB803DEC-438B-43A9-B906-FD7C3B9A0138.svs', 
'TCGA-D8-A3Z6-01Z-00-DX2.19C6AA07-5D58-46CD-91D0-90DD5CC84022.svs', 'TCGA-D8-A1XF-01Z-00-DX2.1460E522-5B87-4690-8B6B-3183C5D282D6.svs', 
'TCGA-A7-A13E-01Z-00-DX1.891954FF-316A-4562-AA14-429631944F22.svs', 'TCGA-D8-A73U-01Z-00-DX1.6ECEF7C0-00CC-4AC2-87BD-DBFB6E0DC042.svs', 
'TCGA-A7-A0CE-01Z-00-DX1.E67322FB-ED25-4B85-B3B0-2B8BD277BB4A.svs', 'TCGA-A7-A13G-01Z-00-DX2.72EF429E-75A7-4D1B-AFFC-8767CB213CDA.svs', 
'TCGA-PL-A8LZ-01A-01-DX1.B9F233EE-06DD-4C14-A392-D2937C7C0868.svs', 'TCGA-A7-A5ZX-01Z-00-DX2.02F586FE-4775-480B-8035-D6AD3386F45D.svs', 
'TCGA-A7-A0CJ-01Z-00-DX1.E26F2F62-D688-4373-BB7B-790A06734E49.svs', 'TCGA-A7-A0CH-01Z-00-DX2.81DDF423-FCA8-46CA-83A4-2E80B611844D.svs', 
'TCGA-D8-A1JE-01Z-00-DX1.714805A1-E337-46DA-88D9-6CE4B4E3C2D0.svs', 'TCGA-D8-A1Y3-01Z-00-DX1.8AA5F695-A06C-4DEA-AD71-16254A48B218.svs', 
'TCGA-D8-A1J9-01Z-00-DX2.E1C59487-9563-4501-845F-2067A0C5C59B.svs', 'TCGA-D8-A141-01Z-00-DX2.DBD0D81E-28FC-4466-BDE3-94753BD6CBEB.svs', 
'TCGA-D8-A1XB-01Z-00-DX1.DA8E2FA4-DBBA-4157-8052-B90FB3BB58F1.svs', 'TCGA-PL-A8LZ-01A-02-DX2.E2697718-12E1-4D7C-9ED8-0A71C6D855B8.svs', 
'TCGA-D8-A13Y-01Z-00-DX1.02321E77-A11E-41A5-95FE-BB897EA5CE58.svs', 'TCGA-D8-A1JC-01Z-00-DX2.854ABF5D-40F1-48AE-802F-97D75497F1FD.svs', 
'TCGA-D8-A1JT-01Z-00-DX1.F278C419-E405-4BDA-BA50-BFBA08801168.svs', 'TCGA-D8-A1Y2-01Z-00-DX2.A563ABFE-18DE-4D78-BB66-9CD18D3CBE3A.svs', 
'TCGA-D8-A1JJ-01Z-00-DX1.a986b48f-b295-4d7a-b778-ce829cdf9c38.svs', 'TCGA-A7-A0DB-01Z-00-DX1.9CE855BC-0C37-43FB-8806-6625E176BE2E.svs', 
'TCGA-D8-A27V-01Z-00-DX1.F937C53B-0B55-4271-843E-2C28F72CF28E.svs', 'TCGA-D8-A145-01Z-00-DX2.B834BF47-1CD6-45EA-BB88-D8ECE1FDDC6A.svs', 
'TCGA-3C-AALJ-01Z-00-DX1.777C0957-255A-42F0-9EEB-A3606BCF0C96.svs', 'TCGA-PL-A8LX-01A-02-DX2.84DA8B75-1196-4704-B6B2-C5E7BF645E95.svs', 
'TCGA-A7-A6VX-01Z-00-DX2.9EE94B59-6A2C-4507-AA4F-DC6402F2B74F.svs', 'TCGA-D8-A1J8-01Z-00-DX2.5DCB3447-548D-442C-86B1-CEF79B8689DF.svs', 
'TCGA-D8-A27T-01Z-00-DX1.1E3A4D57-9CF2-4EBF-B74D-ADD7BD8CBFA5.svs', 'TCGA-A7-A0D9-01Z-00-DX1.FBC3B90F-C58B-4476-8354-0AF9248324E3.svs', 
'TCGA-D8-A73X-01Z-00-DX1.5F0DF75C-594C-42DF-BE3F-E00E5E01DCD6.svs', 'TCGA-D8-A1X6-01Z-00-DX1.ABF237D4-708C-46A5-AEF8-58712E5DCC04.svs', 
'TCGA-A7-A13F-01Z-00-DX2.8CC23BAD-B3CC-4BBC-832A-A3879D6EF62D.svs', 'TCGA-D8-A1XZ-01Z-00-DX1.8E51A61D-B01C-4A52-8F5D-44D2ABCA46FC.svs', 
'TCGA-D8-A140-01Z-00-DX2.0C0A62BB-1FB8-47D8-8FAF-112D221F18BE.svs', 'TCGA-D8-A1JD-01Z-00-DX1.6D215B14-DD90-4635-8645-AF06EBD9BA3F.svs', 
'TCGA-A7-A0DA-01Z-00-DX2.90C93176-C3C6-41B3-B34B-B16F1A1779E6.svs', 'TCGA-A7-A6VV-01Z-00-DX2.4C2BF8C1-CC84-4A6E-BC0F-430BC8BE6B26.svs', 
'TCGA-OK-A5Q2-01Z-00-DX2.C828A160-87DF-4625-A8C5-2057F61D54F4.svs', 'TCGA-D8-A1Y0-01Z-00-DX1.10F40197-4174-43CC-AAD3-8CB85154FB2D.svs',
 'TCGA-D8-A1X7-01Z-00-DX2.F0631B8C-EB75-4995-8ED7-1A8972BE8997.svs']


 # list of patient_ids to delete   # 163 patients
patient_list_to_delete = ['TCGA-E9-A228', 'TCGA-EW-A1OV', 'TCGA-OL-A66O', 'TCGA-GM-A2DD', 'TCGA-E9-A6HE', 'TCGA-EW-A1P6', 'TCGA-EW-A1PA', 
 'TCGA-EW-A1J5', 'TCGA-GM-A2DI', 'TCGA-UL-AAZ6', 'TCGA-EW-A1IY', 'TCGA-EW-A1J1', 'TCGA-LL-A5YL', 'TCGA-E9-A24A', 'TCGA-E9-A248', 
 'TCGA-E9-A2JS', 'TCGA-EW-A1IX', 'TCGA-S3-A6ZF', 'TCGA-OL-A5D6', 'TCGA-LL-A6FP', 'TCGA-A2-A0CX', 'TCGA-S3-A6ZH', 'TCGA-S3-AA11', 
 'TCGA-GM-A2D9', 'TCGA-BH-A1FH', 'TCGA-HN-A2NL', 'TCGA-BH-A1FR', 'TCGA-EW-A1P5', 'TCGA-Z7-A8R5', 'TCGA-A2-A1G6', 'TCGA-LL-A5YN', 
 'TCGA-A2-A0YE', 'TCGA-E9-A22A', 'TCGA-XX-A899', 'TCGA-LL-A8F5', 'TCGA-LD-A7W5', 'TCGA-E9-A5FK', 'TCGA-GM-A2DK', 'TCGA-5L-AAT1', 
 'TCGA-LD-A74U', 'TCGA-LL-A50Y', 'TCGA-EW-A1OX', 'TCGA-D8-A1JM', 'TCGA-LL-A7T0', 'TCGA-EW-A1P4', 'TCGA-EW-A1P8', 'TCGA-GM-A2DN', 
 'TCGA-EW-A2FS', 'TCGA-LL-A5YP', 'TCGA-S3-AA12', 'TCGA-PL-A8LX', 'TCGA-A8-A09V', 'TCGA-PL-A8LY', 'TCGA-GM-A2DL', 'TCGA-OL-A66J', 
 'TCGA-LL-A440', 'TCGA-LD-A7W6', 'TCGA-LL-A73Z', 'TCGA-OL-A66K', 'TCGA-EW-A1P0', 'TCGA-WT-AB44', 'TCGA-OL-A66N', 'TCGA-E9-A243', 
 'TCGA-E9-A247', 'TCGA-EW-A1P7', 'TCGA-E9-A3X8', 'TCGA-EW-A1PB', 'TCGA-A8-A09W', 'TCGA-EW-A1J3', 'TCGA-PE-A5DD', 'TCGA-EW-A6SC', 
 'TCGA-OL-A5D7', 'TCGA-E9-A295', 'TCGA-GM-A2DM', 'TCGA-E9-A229', 'TCGA-S3-AA0Z', 'TCGA-PE-A5DC', 'TCGA-LL-A6FR', 'TCGA-BH-A0EB', 
 'TCGA-LL-A7SZ', 'TCGA-PE-A5DE', 'TCGA-AO-A03P', 'TCGA-E9-A2JT', 'TCGA-OL-A6VQ', 'TCGA-E9-A244', 'TCGA-XX-A89A', 'TCGA-AR-A0TW', 
 'TCGA-GM-A3XL', 'TCGA-EW-A1IZ', 'TCGA-LL-A5YO', 'TCGA-EW-A6SB', 'TCGA-EW-A1P3', 'TCGA-E9-A3HO', 'TCGA-EW-A423', 'TCGA-AR-A0U0', 
 'TCGA-A2-A3XW', 'TCGA-C8-A12V', 'TCGA-GM-A4E0', 'TCGA-A8-A08S', 'TCGA-LL-A740', 'TCGA-LL-A73Y', 'TCGA-OL-A6VO', 'TCGA-EW-A1P1', 
 'TCGA-OL-A66I', 'TCGA-EW-A6SD', 'TCGA-A8-A06X', 'TCGA-EW-A1J2', 'TCGA-EW-A2FR', 'TCGA-B6-A0RQ', 'TCGA-LL-A6FQ', 'TCGA-OL-A5DA', 
 'TCGA-E9-A22H', 'TCGA-OL-A66L', 'TCGA-BH-A0RX', 'TCGA-E9-A22E', 'TCGA-GM-A2DO', 'TCGA-OL-A66P', 'TCGA-LD-A9QF', 'TCGA-E9-A3QA', 
 'TCGA-E9-A22D', 'TCGA-A2-A0ST', 'TCGA-EW-A6S9', 'TCGA-Z7-A8R6', 'TCGA-UU-A93S', 'TCGA-S3-AA10', 'TCGA-GM-A2DC', 'TCGA-GM-A2DB', 
 'TCGA-EW-A1OZ', 'TCGA-WT-AB41', 'TCGA-AR-A24T', 'TCGA-PL-A8LZ', 'TCGA-EW-A1PH', 'TCGA-A2-A0YJ', 'TCGA-OK-A5Q2', 'TCGA-EW-A2FW', 
 'TCGA-HN-A2OB', 'TCGA-GM-A3NW', 'TCGA-A8-A08F', 'TCGA-W8-A86G', 'TCGA-EW-A6SA', 'TCGA-GM-A2DF', 'TCGA-S3-AA17', 'TCGA-B6-A0X1', 
 'TCGA-OL-A5D8', 'TCGA-EW-A1J6', 'TCGA-EW-A1PD', 'TCGA-GM-A2DH', 'TCGA-E9-A54Y', 'TCGA-A8-A08P', 'TCGA-EW-A1IW', 'TCGA-S3-AA14', 
 'TCGA-LL-A9Q3', 'TCGA-GI-A2C8', 'TCGA-AO-A1KO', 'TCGA-S3-A6ZG', 'TCGA-GM-A2DA', 'TCGA-S3-AA15', 'TCGA-E9-A249', 'TCGA-E9-A22G', 
 'TCGA-EW-A1PG', 'TCGA-JL-A3YX', 'TCGA-LD-A66U', 'TCGA-EW-A1OY']
    



def remove_slide_ids(df, name, case_ids_to_delete):
    """
    This function removes specific slide_ids from the datasets_csv files.
    """
    #print(f'#  info pre changes for {name}  #')
    print(f'Number of rows in {name} before changes: {df.shape[0]}')

    # remove the case_ids (the whole row) from the label_data
    if name == 'omics_data':
        idx = df[df.index.isin(case_ids_to_delete)].index.tolist()
    else:
        idx = df[df['case_id'].isin(case_ids_to_delete)].index.tolist()
    print(f'Number of rows to delete from {name}: {len(idx)}')
    #print(f'Rows to delete from {name}: {idx}')

    df = df.drop(idx)
    print(f'Number of rows in {name} after changes: {df.shape[0]}')
    #if len(idx) > 0 and name != 'omics_data':
        #print(f'rows around idx: {df.iloc[(idx[0]-3):(idx[0]+2)]}')

    if name != 'omics_data':
        df = df.reset_index(drop=True)
        #if len(idx) > 0:
            #print(f'rows around idx after index reset: {df.iloc[(idx[0]-3):(idx[0]+2)]}')

    # remove 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        #print(f'columns in {name} after removing Unnamed: 0: {df.columns.tolist()}')
    #print(df.head(5))

    return df

    


def clean_datasets():
    print('Running clean_datasets.py')

    # get the datasets from datasets_csv where the slide_id is mentioned to remove the case_id - remove the patient
    # omics data
    omics_data = pd.read_csv(
        os.path.join('./datasets_csv/raw_rna_data/combine/brca', 'rna_clean.csv'),    # all rna data
        engine='python', 
        index_col=0
    )

    # metadata
    label_data = pd.read_csv('./datasets_csv/metadata/tcga_brca.csv', low_memory=False)    # contains pairing between case_id and slide_id(s) + info on survival

    # clinical data
    clinical_data = pd.read_csv('./datasets_csv/clinical_data/tcga_brca_clinical.csv', index_col=0) # clinical data

    # now get the case_id corresponding to the slide_ids in the list_to_delete
    case_ids_to_delete = label_data[label_data['slide_id'].isin(list_to_delete)]['case_id'].tolist()
    print(f'Number of case_ids to delete: {len(case_ids_to_delete)}')
    #print(f'Case_ids to delete: {case_ids_to_delete}')

    # add the patients not present in wsi database
    case_ids_to_delete.extend(patient_list_to_delete)

    # remove the case_ids from the datasets
    omics_data = remove_slide_ids(omics_data, 'omics_data', case_ids_to_delete)
    label_data = remove_slide_ids(label_data, 'label_data', case_ids_to_delete)
    clinical_data = remove_slide_ids(clinical_data, 'clinical_data', case_ids_to_delete)

    #save changes
    omics_data.to_csv(os.path.join('./datasets_csv/raw_rna_data/combine/brca', 'rna_clean.csv'))
    label_data.to_csv('./datasets_csv/metadata/tcga_brca.csv')
    clinical_data.to_csv('./datasets_csv/clinical_data/tcga_brca_clinical.csv')



def clean_datasets_label(label_data):
    print('##### clean label_data #####')

    # now get the case_id corresponding to the slide_ids in the list_to_delete
    case_ids_to_delete = label_data[label_data['slide_id'].isin(list_to_delete)]['case_id'].tolist()
    case_ids_to_delete.extend(patient_list_to_delete)
    print(f'Number of case_ids to delete: {len(case_ids_to_delete)}')
    #print(f'Case_ids to delete: {case_ids_to_delete}')

    # remove the case_ids from the dataset
    label_data = remove_slide_ids(label_data, 'label_data', case_ids_to_delete)
    return label_data

def clean_datasets_omics(omics_data):

    print('##### clean omics_data #####')
    case_ids_to_delete = []
    for wsi in list_to_delete:
        case_id = wsi[:12]
        case_ids_to_delete.append(case_id)
    case_ids_to_delete.extend(patient_list_to_delete)

    # remove the case_ids from the dataset
    omics_data = remove_slide_ids(omics_data, 'omics_data', case_ids_to_delete)
    return omics_data

def clean_datasets_clinical(clinical_data):
    
    print('##### clean clinical_data #####')

    case_ids_to_delete = []
    for wsi in list_to_delete:
        case_id = wsi[:12]
        case_ids_to_delete.append(case_id)
    case_ids_to_delete.extend(patient_list_to_delete)

    # remove the case_ids from the dataset
    clinical_data = remove_slide_ids(clinical_data, 'clinical_data', case_ids_to_delete)
    return clinical_data