import os

case_id_col = {}
activity_col = {}
resource_col = {}
timestamp_col = {}
label_col = {}
pos_label = {}
neg_label = {}
dynamic_cat_cols = {}
static_cat_cols = {}
dynamic_num_cols = {}
static_num_cols = {}
filename = {}

logs_dir = "/storage/hpc_irheta/labeled_logs_csv_processed"
#logs_dir = "/storage/hpc_irheta/labeled_logs_csv_processed_resource"

#### BPIC2011 settings ####
for formula in range(1,5):
    dataset = "bpic2011_f%s"%formula 
    
    filename[dataset] = os.path.join(logs_dir, "BPIC11_f%s.csv"%formula)
    
    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity code"
    resource_col[dataset] = "Producer code"
    timestamp_col[dataset] = "Complete Timestamp"
    label_col[dataset] = "label"
    pos_label[dataset] = "deviant"
    neg_label[dataset] = "regular"

    # features for classifier
    dynamic_cat_cols[dataset] = ["Activity code", "Producer code", "Section", "Specialism code", "group"]
    static_cat_cols[dataset] = ["Diagnosis", "Treatment code", "Diagnosis code", "case Specialism code", "Diagnosis Treatment Combination ID"]
    dynamic_num_cols[dataset] = ["Number of executions", "duration", "month", "weekday", "hour"]
    static_num_cols[dataset] = ["Age"]
    

    
#### BPIC2015 settings ####
for municipality in range(1,6):
    for formula in range(1,3):
        dataset = "bpic2015_%s_f%s"%(municipality, formula)
        
        filename[dataset] = os.path.join(logs_dir, "BPIC15_%s_f%s.csv"%(municipality, formula))

        case_id_col[dataset] = "Case ID"
        activity_col[dataset] = "Activity"
        resource_col[dataset] = "Resource"
        timestamp_col[dataset] = "Complete Timestamp"
        label_col[dataset] = "label"
        pos_label[dataset] = "deviant"
        neg_label[dataset] = "regular"

        # features for classifier
        dynamic_cat_cols[dataset] = ["Activity", "monitoringResource", "question", "Resource"]
        static_cat_cols[dataset] = ["Responsible_actor"]
        dynamic_num_cols[dataset] = ["duration", "month", "weekday", "hour"]
        static_num_cols[dataset] = ["SUMleges", 'Aanleg (Uitvoeren werk of werkzaamheid)', 'Bouw', 'Brandveilig gebruik (vergunning)', 'Gebiedsbescherming', 'Handelen in strijd met regels RO', 'Inrit/Uitweg', 'Kap', 'Milieu (neutraal wijziging)', 'Milieu (omgevingsvergunning beperkte milieutoets)', 'Milieu (vergunning)', 'Monument', 'Reclame', 'Sloop']
        
        if municipality in [3,5]:
            static_num_cols[dataset].append('Flora en Fauna')
        if municipality in [1,2,3,5]:
            static_num_cols[dataset].append('Brandveilig gebruik (melding)')
            static_num_cols[dataset].append('Milieu (melding)')

        
        
#### Traffic fines settings ####
dataset = "traffic_fines"

filename[dataset] = os.path.join(logs_dir, "Road_Traffic_Fine_Management_Process.csv")

case_id_col[dataset] = "Case ID"
activity_col[dataset] = "Activity"
timestamp_col[dataset] = "Complete Timestamp"
label_col[dataset] = "label"
pos_label[dataset] = "deviant"
neg_label[dataset] = "regular"

# features for classifier
dynamic_cat_cols[dataset] = ["Activity", "Resource", "lastSent", "notificationType", "dismissal"]
static_cat_cols[dataset] = ["article", "vehicleClass"]
dynamic_num_cols[dataset] = ["expense", "duration", "month", "weekday", "hour"]
static_num_cols[dataset] = ["amount", "points"]


for formula in range(1,3):
    dataset = "traffic_fines_%s"%formula
    
    filename[dataset] = os.path.join(logs_dir, "traffic_fines_%s.csv"%formula)
    
    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = "Resource"
    timestamp_col[dataset] = "Complete Timestamp"
    label_col[dataset] = "label"
    pos_label[dataset] = "deviant"
    neg_label[dataset] = "regular"

    # features for classifier
    dynamic_cat_cols[dataset] = ["Activity", "Resource", "lastSent", "notificationType", "dismissal"]
    static_cat_cols[dataset] = ["article", "vehicleClass"]
    dynamic_num_cols[dataset] = ["expense", "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour", "open_cases"]
    static_num_cols[dataset] = ["amount", "points"]
        

#### Sepsis Cases settings ####
datasets = ["sepsis_cases_%s" % i for i in range(1, 5)]

for dataset in datasets:
    
    filename[dataset] = os.path.join(logs_dir, "%s.csv" % dataset)

    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = "org:group"
    timestamp_col[dataset] = "time:timestamp"
    label_col[dataset] = "label"
    pos_label[dataset] = "deviant"
    neg_label[dataset] = "regular"

    # features for classifier
    dynamic_cat_cols[dataset] = ["Activity", 'org:group'] # i.e. event attributes
    static_cat_cols[dataset] = ['Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
                       'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
                       'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
                       'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg',
                       'Hypotensie', 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie',
                       'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea',
                       'SIRSCritTemperature', 'SIRSCriteria2OrMore'] # i.e. case attributes that are known from the start
    dynamic_num_cols[dataset] = ['CRP', 'LacticAcid', 'Leucocytes', "hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart"]#, "event_nr", "open_cases"]
    static_num_cols[dataset] = ['Age']


#### Production log settings ####
dataset = "production"

filename[dataset] = os.path.join(logs_dir, "Production.csv")

case_id_col[dataset] = "Case ID"
activity_col[dataset] = "Activity"
resource_col[dataset] = "Resource"
timestamp_col[dataset] = "Complete Timestamp"
label_col[dataset] = "label"
neg_label[dataset] = "regular"
pos_label[dataset] = "deviant"

# features for classifier
static_cat_cols[dataset] = ["Part_Desc_", "Rework"]
static_num_cols[dataset] = ["Work_Order_Qty"]
dynamic_cat_cols[dataset] = ["Activity", "Resource", "Report_Type", "Resource.1"]
dynamic_num_cols[dataset] = ["Qty_Completed", "Qty_for_MRB", "activity_duration", "hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]

#### BPIC2017 settings ####

bpic2017_dict = {"bpic2017_cancelled": "BPIC17_O_Cancelled.csv",
                 "bpic2017_accepted": "BPIC17_O_Accepted.csv",
                 "bpic2017_refused": "BPIC17_O_Refused.csv",
                 "bpic2017_cancelled_complete": "BPIC17_O_Cancelled_complete.csv",
                 "bpic2017_accepted_complete": "BPIC17_O_Accepted_complete.csv",
                 "bpic2017_refused_complete": "BPIC17_O_Refused_complete.csv"
                }

for dataset, fname in bpic2017_dict.items():

    filename[dataset] = os.path.join(logs_dir, fname)

    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = 'org:resource'
    timestamp_col[dataset] = 'time:timestamp'
    label_col[dataset] = "label"
    neg_label[dataset] = "regular"
    pos_label[dataset] = "deviant"

    # features for classifier
    dynamic_cat_cols[dataset] = ["Activity", 'org:resource', 'Action', 'EventOrigin', 'lifecycle:transition',
                                "Accepted", "Selected"] 
    static_cat_cols[dataset] = ['ApplicationType', 'LoanGoal']
    dynamic_num_cols[dataset] = ['FirstWithdrawalAmount', 'MonthlyCost', 'NumberOfTerms', 'OfferedAmount', 'CreditScore',  "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour", "open_cases"]
    static_num_cols[dataset] = ['RequestedAmount']
    
    
#### BPIC2017 without open cases settings ####

bpic2017_dict = {"bpic2017_cancelled_wo_open_cases": "BPIC17_O_Cancelled_wo_open_cases.csv",
                 "bpic2017_accepted_wo_open_cases": "BPIC17_O_Accepted_wo_open_cases.csv",
                 "bpic2017_refused_wo_open_cases": "BPIC17_O_Refused_wo_open_cases.csv"
                }

for dataset, fname in bpic2017_dict.items():

    filename[dataset] = os.path.join(logs_dir, fname)

    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = 'org:resource'
    timestamp_col[dataset] = 'time:timestamp'
    label_col[dataset] = "label"
    neg_label[dataset] = "regular"
    pos_label[dataset] = "deviant"

    # features for classifier
    dynamic_cat_cols[dataset] = ["Activity", 'org:resource', 'Action', 'EventOrigin', 'lifecycle:transition',
                                "Accepted", "Selected", 'CreditScore'] 
    static_cat_cols[dataset] = ['ApplicationType', 'LoanGoal']
    dynamic_num_cols[dataset] = ['FirstWithdrawalAmount', 'MonthlyCost', 'NumberOfTerms', 'OfferedAmount',  "timesincelastevent", "timesincecasestart", "event_nr", "month", "weekday", "hour"]
    static_num_cols[dataset] = ['RequestedAmount']


#### Hospital billing settings ####
for i in range(1, 7):
    for suffix in ["", "_sample10000", "_sample30000"]:
        dataset = "hospital_billing_%s%s" % (i, suffix)

        filename[dataset] = os.path.join(logs_dir, "hospital_billing_%s%s.csv" % (i, suffix))

        case_id_col[dataset] = "Case ID"
        activity_col[dataset] = "Activity"
        resource_col[dataset] = "Resource"
        timestamp_col[dataset] = "Complete Timestamp"
        label_col[dataset] = "label"
        neg_label[dataset] = "regular"
        pos_label[dataset] = "deviant"

        if i == 1:
            neg_label[dataset] = "deviant"
            pos_label[dataset] = "regular"

        # features for classifier
        dynamic_cat_cols[dataset] = ["Activity", 'Resource', 'actOrange', 'actRed', 'blocked', 'caseType', 'diagnosis', 'flagC', 'flagD', 'msgCode', 'msgType', 'state', 'version']#, 'isCancelled', 'isClosed', 'closeCode'] 
        static_cat_cols[dataset] = ['speciality']
        dynamic_num_cols[dataset] = ['msgCount', "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour", "open_cases"]
        static_num_cols[dataset] = []

        if i == 1: # label is created based on isCancelled attribute
            dynamic_cat_cols[dataset] = [col for col in dynamic_cat_cols[dataset] if col != "isCancelled"]
        elif i == 2:
            dynamic_cat_cols[dataset] = [col for col in dynamic_cat_cols[dataset] if col != "isClosed"]
            
            
#### BPIC2012 settings ####
bpic2012_dict = {"bpic2012_cancelled": "bpic2012_O_CANCELLED-COMPLETE.csv",
                 "bpic2012_accepted": "bpic2012_O_ACCEPTED-COMPLETE.csv",
                 "bpic2012_declined": "bpic2012_O_DECLINED-COMPLETE.csv",
                 "bpic2012_cancelled_complete": "bpic2012_O_CANCELLED-COMPLETE_complete.csv",
                 "bpic2012_accepted_complete": "bpic2012_O_ACCEPTED-COMPLETE_complete.csv",
                 "bpic2012_declined_complete": "bpic2012_O_DECLINED-COMPLETE_complete.csv"
                }

for dataset, fname in bpic2012_dict.items():

    filename[dataset] = os.path.join(logs_dir, fname)

    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = "Resource"
    timestamp_col[dataset] = "Complete Timestamp"
    label_col[dataset] = "label"
    neg_label[dataset] = "regular"
    pos_label[dataset] = "deviant"

    # features for classifier
    dynamic_cat_cols[dataset] = ["Activity", "Resource"]
    static_cat_cols[dataset] = []
    dynamic_num_cols[dataset] = ["hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]
    static_num_cols[dataset] = ['AMOUNT_REQ']
    

#### BPIC2013 settings ####
bpic2013_dict = {"bpic2013_1": "VINST cases incidents_1.csv",
                 "bpic2013_2": "VINST cases incidents_2.csv",
                 "bpic2013_3": "VINST cases incidents_3.csv"
                }

for dataset, fname in bpic2013_dict.items():

    filename[dataset] = os.path.join(logs_dir, fname)

    case_id_col[dataset] = "SR Number"
    timestamp_col[dataset] = "Change Date+Time"
    activity_col[dataset] = "Sub Status"
    resource_col[dataset] = "Involved ST"
    label_col[dataset] = "label"
    neg_label[dataset] = "regular"
    pos_label[dataset] = "deviant"

    # features for classifier
    dynamic_cat_cols[dataset] = ["Sub Status", "Involved ST", 'Status', 'Involved ST Function Div', 'Involved Org line 3',
                   "Owner Country", "Owner First Name"] # i.e. event attributes
    static_cat_cols[dataset] = ["Product", "Country", "SR Latest Impact"] # i.e. case attributes that are known from the start
    dynamic_num_cols[dataset] = ["timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour", "open_cases"]
    static_num_cols[dataset] = []
    

#### Minit invoice approval log settings ####
dataset = "minit"

filename[dataset] = os.path.join(logs_dir, "Minit_invoice_approval.csv")

case_id_col[dataset] = "Case ID"
activity_col[dataset] = "Activity"
resource_col[dataset] = "Resource"
timestamp_col[dataset] = "Complete Timestamp"
label_col[dataset] = "label"
neg_label[dataset] = "regular"
pos_label[dataset] = "deviant"

# features for classifier
static_cat_cols[dataset] = ["CostCenter.Code", "Supplier.City", "Supplier.Name", "Supplier.State"]# "InvoiceStatus.DisplayName"]
static_num_cols[dataset] = ["InvoiceTotalAmountWithoutVAT"]
dynamic_cat_cols[dataset] = ["Activity", "Resource", "ActivityFinalAction", "EventType"]
dynamic_num_cols[dataset] = ["hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr"]


#### Helpdesk settings ####
dataset = "helpdesk" 

filename[dataset] = os.path.join(logs_dir, "helpdesk.csv")
    
case_id_col[dataset] = "CaseID"
activity_col[dataset] = "ActivityID"
timestamp_col[dataset] = "CompleteTimestamp"
label_col[dataset] = None
pos_label[dataset] = None
neg_label[dataset] = None

# features for classifier
dynamic_cat_cols[dataset] = ["ActivityID"]
static_cat_cols[dataset] = []
dynamic_num_cols[dataset] = []
static_num_cols[dataset] = []


#### BPIC2012 W settings ####
dataset = "bpic2012_w" 

filename[dataset] = os.path.join(logs_dir, "bpi_12_w.csv")
    
case_id_col[dataset] = "CaseID"
activity_col[dataset] = "ActivityID"
timestamp_col[dataset] = "CompleteTimestamp"
label_col[dataset] = None
pos_label[dataset] = None
neg_label[dataset] = None

# features for classifier
dynamic_cat_cols[dataset] = ["ActivityID"]
static_cat_cols[dataset] = []
dynamic_num_cols[dataset] = []
static_num_cols[dataset] = []
