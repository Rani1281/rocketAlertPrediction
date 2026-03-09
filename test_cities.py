import pandas as pd

# cities_df = pd.read_csv('data/raw/cities.csv')
# cities = cities_df['City']
#
# example_city_alert = 'באר שבע - צפון'
# result_df = cities_df[cities_df['City'].apply(lambda x: isinstance(x, str) and x in example_city_alert)]
# if not result_df.empty:
#     result_city = result_df['City'].iat[0]
#     print(result_city)

alerts_df = pd.read_csv('data/raw/israel_alerts.csv')
print(alerts_df['data'])



