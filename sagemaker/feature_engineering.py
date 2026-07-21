import os
import pandas as pd
import numpy as np
import datetime
from dateutil.tz import tzlocal, tzwinlocal


INPUT_FILE = "/opt/ml/processing/input/data.xlsx"
OUTPUT_FILE = "/opt/ml/processing/output/"


def filter_patching_instances(df):
    df = df.copy()

    df = df[
        (df["PatchGroup"].isin(
            ["sat_22:00", "sat_22:00_rollingreboot"]
        ))]

    return df


def patch_group_instances_feature(df):
    feature_df = (
        df.groupby("AccountId")["ResourceId"]
        .nunique()
        .reset_index(name="Patch_Group_Instances")
        )
    feature_df["AccountId"] = feature_df["AccountId"].astype(str)
    return feature_df


def stop_instance_pct_feature(df):
    df["EC2-State"] = df["EC2-State"].str.lower().str.strip()
    feature_df = (
        df.groupby("AccountId")
        .agg(
            Stopped_Instances=("EC2-State", lambda x: (x == "stopped").sum()),
            Running_Instances=("EC2-State", lambda x: (x == "running").sum())
        )
        .reset_index()
    )
    feature_df["AccountId"] = feature_df["AccountId"].astype(str)
    patch_group_instances_feature_df = patch_group_instances_feature(df)
    feature_df["Stopped_Instances_Percentage"] = (
        feature_df["Stopped_Instances"] /
        patch_group_instances_feature_df["Patch_Group_Instances"] * 100
    ).round(2)

    return feature_df


def ssm_ping_status_features(df):
    # Replace NaN with blank
    df["ssm_ping_status"] = df["ssm_ping_status"].fillna("")

    # Remove leading/trailing spaces
    df["ssm_ping_status"] = df["ssm_ping_status"].astype(str).str.strip().str.lower()

    # Create Offline Flag
    # Offline means:
    # 1. connectionlost
    # 2. blank status
    df["Offline"] = df["ssm_ping_status"].isin(["connectionlost", ""])
    patch_group_instances_feature_df = patch_group_instances_feature(df)
    # Group account-wise
    feature_df = (
        df.groupby("AccountId")
        .agg(
            Offline_Instances=("Offline", "sum"),
            Online_Instances=("ssm_ping_status", lambda x: (x == "online").sum())
        )
        .reset_index()
    )
    feature_df["AccountId"] = feature_df["AccountId"].astype(str)

    # Calculate Percentage
    feature_df["Offline_Instances_Percentage"] = (
        feature_df["Offline_Instances"]
        / patch_group_instances_feature_df["Patch_Group_Instances"]
        * 100
    ).round(2)

    return feature_df


def total_noncompliantcount_feature(df):
    df["NonCompliance_Log"] = np.log1p(pd.to_numeric(df["TotalNonCompliantCount"], errors="coerce"))
    # Handle missing values
    df["TotalNonCompliantCount"] = (df["TotalNonCompliantCount"].fillna(0))
    #account_id = "202347608077"
    #account_data = df[df["AccountId"] == account_id]
    #print(account_data["NonCompliance_Log"].sum())  # Print the sum of NonCompliance_Log for the specific account for debugging
    #print(df["NonCompliance_Log"].tail(100))  # Print the last 10 values of the NonCompliance_Log column for debugging
    feature_df = (
        df.groupby("AccountId")
        .agg(
            Total_NonCompliance_Log_Score=(
                "NonCompliance_Log",
                "sum"
            )
        ).reset_index()
    )
    feature_df["AccountId"] = feature_df["AccountId"].astype(str)
    return feature_df


def installed_pending_reboot_percentage(df):
    df = df.copy()
    patch_group_instances_feature_df = patch_group_instances_feature(df)
    # Convert blank values to empty list
    df["Pending_Reboot_Flag"] = (
        ~df["InstalledPendingReboot"]
        .fillna("")
        .astype(str)
        .str.strip()
        .isin(["", "[]"])
    ).astype(int)

    feature_df = (
        df.groupby("AccountId").agg(
            Pending_Reboot_Instances=(
                "Pending_Reboot_Flag",
                "sum"
            )
        ).reset_index()
    )

    feature_df["Pending_Reboot_Percentage"] = (
        feature_df["Pending_Reboot_Instances"]
        /
        patch_group_instances_feature_df["Patch_Group_Instances"]
        * 100
    ).round(2)

    return feature_df


def total_pending_reboot_patch_count_feature(df):
    df = df.copy()

    # Convert InstalledPendingReboot column into Python list
    def convert_to_list(value):
        if pd.isna(value):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            if value.strip() == "[]":
                return []
            try:
                return eval(
                    value,
                    {
                        "datetime": datetime,
                        "tzlocal": tzlocal
                    }
                )
            except Exception as e:
                print("Conversion failed:")
                print(value)
                print("Error:", e)
                return []
        return []

    df["InstalledPendingReboot"] = (
        df["InstalledPendingReboot"]
        .apply(convert_to_list)
    )

    # Count InstalledPendingReboot patches per EC2 instance
    def count_pending_reboot_patches(patch_list):
        print("Patch list:", patch_list)  # Debugging line
        if not isinstance(patch_list, list):
            return 0
        count = 0
        for patch in patch_list:
            if isinstance(patch, dict):
                state = str(
                    patch.get("State", "")
                ).strip().lower()
                if state == "installedpendingreboot":
                    count += 1
        return count

    df["Pending_Reboot_Patch_Count"] = (
        df["InstalledPendingReboot"]
        .apply(count_pending_reboot_patches)
    )

    # Account wise aggregation
    feature_df = (
        df.groupby("AccountId")
        .agg(
            Total_Pending_Reboot_Patch_Count=(
                "Pending_Reboot_Patch_Count",
                "sum"
            )
        )
        .reset_index()
    )

    feature_df["AccountId"] = (feature_df["AccountId"].astype(str))
    return feature_df


def total_failed_patch_count_feature(df):
    df = df.copy()

    # Convert FailedPatch column into Python list
    def convert_to_list(value):

        if pd.isna(value):
            return []

        if isinstance(value, list):
            return value

        if isinstance(value, str):

            if value.strip() == "[]":
                return []

            try:
                return eval(
                    value,
                    {
                        "datetime": datetime,
                        "tzlocal": tzlocal,
                        "tzwinlocal": tzwinlocal
                    }
                )

            except Exception as e:
                print("Conversion failed:")
                print(value)
                print("Error:", e)
                return []

        return []

    df["FailedPatch"] = (df["FailedPatch"].apply(convert_to_list))

    # Count FailedPatch patches per EC2 instance
    def count_failed_patch_patches(patch_list):
        print("Patch list:", patch_list)  # Debugging line
        if not isinstance(patch_list, list):
            return 0
        count = 0
        for patch in patch_list:
            if isinstance(patch, dict):
                state = str(
                    patch.get("State", "")
                ).strip().lower()
                if state == "failed":
                    count += 1
        return count

    df["Failed_Patch_Patch_Count"] = (df["FailedPatch"].apply(count_failed_patch_patches))

    # Account wise aggregation
    feature_df = (
        df.groupby("AccountId")
        .agg(
            Total_Failed_Patch_Patch_Count=(
                "Failed_Patch_Patch_Count",
                "sum"
            )
        )
        .reset_index()
    )

    feature_df["AccountId"] = (feature_df["AccountId"].astype(str))
    return feature_df


def total_missing_patch_count_feature(df):
    df = df.copy()

    # Convert MissingPatch column into Python list
    def convert_to_list(value):

        if pd.isna(value):
            return []

        if isinstance(value, list):
            return value

        if isinstance(value, str):

            if value.strip() == "[]":
                return []

            try:
                return eval(
                    value,
                    {
                        "datetime": datetime,
                        "tzlocal": tzlocal,
                        "tzwinlocal": tzwinlocal
                    }
                )

            except Exception as e:
                print("Conversion failed:")
                print(value)
                print("Error:", e)
                return []

        return []

    df["MissingPatch"] = (df["MissingPatch"].apply(convert_to_list))

    # Count MissingPatch patches per EC2 instance
    def count_missing_patches(patch_list):
        print("Patch list:", patch_list)  # Debugging line
        if not isinstance(patch_list, list):
            return 0
        count = 0
        for patch in patch_list:
            if isinstance(patch, dict):
                state = str(
                    patch.get("State", "")
                ).strip().lower()
                if state == "missing":
                    count += 1
        return count

    df["Missing_Patch_Count"] = (df["MissingPatch"].apply(count_missing_patches))

    # Account wise aggregation
    feature_df = (
        df.groupby("AccountId")
        .agg(
            Total_Missing_Patch_Count=(
                "Missing_Patch_Count",
                "sum"
            )
        )
        .reset_index()
    )

    feature_df["AccountId"] = (feature_df["AccountId"].astype(str))
    return feature_df


def os_type_percentage_feature(df):
    patch_group_instances_feature_df = patch_group_instances_feature(df)
    feature_df = df.groupby("AccountId").agg(
        Windows_Count=("PlatformDetails", lambda x: x.str.contains("Windows", case=False, na=False).sum()),
        Linux_Count=("PlatformDetails", lambda x: x.str.contains("Linux", case=False, na=False).sum()),
    ).reset_index()
    
    feature_df["Windows_Percentage"] = (
        feature_df["Windows_Count"] / patch_group_instances_feature_df["Patch_Group_Instances"]
    * 100 ).round(2)
    feature_df["Linux_Percentage"] = (
        feature_df["Linux_Count"] / patch_group_instances_feature_df["Patch_Group_Instances"]
    * 100 ).round(2)
    return feature_df


def old_instance_age_feature(df):
    df = df.copy()
    patch_group_instances_feature_df = patch_group_instances_feature(df)
    # Keep only rows containing "days"
    df = df[
        df["Uptime"]
        .astype(str)
        .str.contains("days", case=False, na=False)
    ].copy()

    # Remove "days" and convert to integer
    df["uptime_days"] = (
        df["Uptime"]
        .str.replace(" days", "", regex=False)
        .astype(int)
    )
    # Flag instances older than 730 days
    df["Old_Instance_Flag"] = (df["uptime_days"] > 365)

    # Account wise aggregation
    feature_df = (
        df.groupby("AccountId")
        .agg(
            Old_Instance_Count=(
                "Old_Instance_Flag",
                "sum"
            )
        )
        .reset_index()
    )

    # Percentage of old instances
    feature_df["Old_Instance_Percentage"] = (
        feature_df["Old_Instance_Count"]
        /
        patch_group_instances_feature_df["Patch_Group_Instances"]
        * 100
    ).round(2)

    feature_df["AccountId"] = (
        feature_df["AccountId"]
        .astype(str)
    )

    return feature_df


def write_features_to_excel(feature_df, output_file):
    output_file = os.path.join(output_file, "features_data.xlsx")
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        feature_df.to_excel(writer, sheet_name="Feature_Data", index=False)
    print("Feature file created successfully.")
# Write output


def create_final_dataset(df):
    feature_df = (
        patch_group_instances_feature(df)
        .merge(stop_instance_pct_feature(df), on="AccountId")
        .merge(ssm_ping_status_features(df), on="AccountId")
        .merge(total_noncompliantcount_feature(df), on="AccountId")
        .merge(installed_pending_reboot_percentage(df), on="AccountId")
        .merge(total_pending_reboot_patch_count_feature(df), on="AccountId", how="left")
        .merge(total_failed_patch_count_feature(df), on="AccountId", how="left")
        .merge(total_missing_patch_count_feature(df), on="AccountId", how="left")
        .merge(os_type_percentage_feature(df), on="AccountId", how="left")
        .merge(old_instance_age_feature(df), on="AccountId", how="left")
    )
    return feature_df


# Makind directory for output if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Find uploaded Excel file
excel_files = [
    f for f in os.listdir(os.path.dirname(INPUT_FILE))
    if f.endswith(".xlsx")
]

if not excel_files:
    raise Exception("No Excel file found")

input_file = os.path.join(os.path.dirname(INPUT_FILE), excel_files[0])
print(f"Reading {input_file}")
master_df = pd.read_excel(input_file)

# Filter patching instances
filtered_df = filter_patching_instances(master_df)

# Write filtered DataFrame to Excel
filtered_df = filtered_df.astype(str)
filtered_df.to_excel(os.path.join(OUTPUT_FILE, "output.xlsx"), index=False)

# Generate features
feature_df = create_final_dataset(filtered_df)
write_features_to_excel(feature_df, OUTPUT_FILE)
