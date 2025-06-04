---
date: '2025-02-13T12:28:23-08:00'
draft: false
title: 'GA to Snowflake'
description: The requirement(s) was to build an Ecommerce dashboard with data sources from Shopify, Google Analytics, and local database. #Omo Portfolio description
cover:
    image: img/GA_SF.png # image path/url
    alt: "This" # alt text
    caption: " " # display caption under cover
tags: ["GA", "Snowflake", "Marketing", "Retail"]
categories: ["Tech"]

---
<div style="text-align: justify; max-width: 700px; margin: auto;">

This project was an interesting one. The tools used for the project are listed below:
- Jupyter 
- Google console (console.cloud.google)
- Ga-dev-tools (ga-dev-tools)
- Snowflake (app.snowflake.com)

I will give a brief summary of the steps involved in the project from start to finish.
- Step 1 : I was to reveiw the front end of these applications and identify the key metrics that were needed.
- Step 2 : After identifying these data sources, a plan was drafted on how to get these data out of the applications. The data was gotten from the shopify api, google analytics api 4 (GA4), oracle, and snowflake databases.
- Step 3 : I connected to the GA4 developers [console](https://console.cloud.google.com/) to create my method of authentication. I used Oauth 2.0 client IDs for my authentication since I didn't have a service account and running this on my pc.     
The basic steps are :
    - Go to the Google Cloud Console (have the property id ready).
    - Navigate to APIs & Services > Credentials.
    - Click on Create Credentials and select OAuth 2.0 Client IDs.
    - Configure the consent screen and download the JSON file containing your client ID and client secret. "Don't forget to Set the default Ips in the redirect URIs 🙂." 
    - Next, run this script below to generate your access token. This will automatically download on the specified path.
</div>

```python
import os
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.analytics.data_v1beta import BetaAnalyticsDataClient

# Path to your OAuth credentials JSON
OAUTH_JSON_PATH = "C:\\Users\\Downloads\\Python\\Oauth_Credentials.json"
TOKEN_PATH = "C:\\Users\\Downloads\\Python\\token.json"

def get_credentials():
    creds = None

    # Load existing credentials if available
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH)

    # If there are no valid credentials, authenticate using OAuth JSON
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())  # Refresh token automatically
        else:
            raise Exception("❌ OAuth credentials have expired or are missing. Please authenticate manually first.")

    # Save updated credentials (with refreshed token)
    with open(TOKEN_PATH, "w") as token_file:
        token_file.write(creds.to_json())

    return creds

# Get authenticated credentials
credentials = get_credentials()
client = BetaAnalyticsDataClient(credentials=credentials)
```
<div style="text-align: justify; max-width: 700px; margin: auto;">
* Note: If yor python environment is having outdated widgets especially using jupyter, this could cause some errors.
- Step 4 : I used the dev tools query explorer to select the dimensions and metrics I needed and ran the request to see the response. 
- Step 5 : I wrote my script and kept modifying until completed. This script connected to snowflake; extracted, transformed some fields, and Loaded data into the already created snowflake table. After these, next is to create a task for daily data update. Amongst the multiple ways to update data, I opted for windows task scheduler. I found this option more convenient considering the resources I have available.
The snowflake connector needs to be installed by running.
</div>

```python
pip install snowflake-connector-python
```
There's also need to setup the following :   
    - User:  
    - Password:  
    - Account:  
    - Warehouse:  
    - Database:  
    - Schema:  

* Note: GA4 default limit is 10000 rows. you can use pagination (limit and offset) to get more rows. 

Here is a snippet of the python code 

``` python
# Google Analytics Property ID
property_id = "12345687"

# Date range Get previous day data after retrieving historical data
start_date = "yesterday"
end_date = "yesterday"

# API request
BATCH_SIZE = 10000
all_data = []
offset = 0

while True:
    request = RunReportRequest(
        property=f"properties/{property_id}",
        date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        metrics=[
            Metric(name="sessions"),
            Metric(name="totalRevenue"),
            Metric(name="transactions"),
            Metric(name="taxAmount")
        ],
        dimensions=[
            Dimension(name="date"),
            Dimension(name="transactionId"),
            Dimension(name="sessionPrimaryChannelGroup"),
            Dimension(name="deviceCategory"),
            Dimension(name="country"),
            Dimension(name="city"),
            Dimension(name="region")
        ],
        limit=BATCH_SIZE,
        offset=offset
    )
    try:
        response = client.run_report(request)
        logging.info(f"Fetched {len(response.rows)} rows starting from offset {offset}.")
# If no rows are returned, break the loop
        if not response.rows:
            break
        # Append the retrieved data
        for row in response.rows:
            all_data.append(
                [value.value for value in row.dimension_values] +
                [float(value.value) if '.' in value.value else int(value.value) for value in row.metric_values]
            )
    # Increase the offset for the next batch
        offset += BATCH_SIZE

    except Exception as e:
        logging.error(f"Error fetching GA data: {e}", exc_info=True)
        break

# Convert the collected data into a DataFrame
columns = [
     "EVENT_DATE", "TRANSACTION_ID", "SESSION_PRIMARY_CHANNEL_GROUP", "DEVICE_CATEGORY",
    "COUNTRY", "CITY", "REGION", "SESSIONS", "TOTAL_REVENUE", "TRANSACTIONS", "TAX_AMOUNT"
]
df = pd.DataFrame(all_data, columns=columns)

# Convert date format
df["EVENT_DATE"] = pd.to_datetime(df["EVENT_DATE"], format="%Y%m%d").dt.strftime("%Y-%m-%d")

# Generate composite key
```

- Step 6 : Get data from shopify using airbyte. For step by step instructions, consult the documentation here [airbyte.com](https://docs.airbyte.com/integrations/sources/shopify/).
- Step 7 : Create data models on snowflake with dimensions and metrics using joins and unions where needed in separate views, connected tableau to the views, designed dashboard, and published to Tableau Online.

The db Schema, tables, and views were created as seen below:

![image](/img/GA4_DB.png#center)

The tableau dashboard draft is seen below:

![image](/img/Ecom_kpi.png#center)

This was tweaked according to user requirement and deployed.