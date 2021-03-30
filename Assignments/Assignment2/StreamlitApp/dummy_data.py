import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gen_journey_data(campaigns = 20, journeys=100, min_len =2, max_len= 10, conversion_rate=0.4):
    # Generate required aggregation
    journey_lengths = np.random.randint(low= min_len, high=max_len,size=journeys)
    journey_summary = pd.DataFrame({
        "jid":np.arange(journeys),
        "jlen":journey_lengths,
        "converted" : np.random.choice([1,0], size=journeys, replace=True, p=[conversion_rate, 1-conversion_rate])
    })

    # Expand the aggregated values into full journeys
    def expand_journeys(row, campaign_size=campaigns):
        result = pd.DataFrame({"jid":np.repeat(row["jid"],row["jlen"]),
                        "conversion" : np.repeat(row["converted"],row["jlen"]),
                        "timestamp_norm": np.random.uniform(size=row["jlen"]),
                        "campaign" : np.random.randint(low=0,high=campaign_size, size=row["jlen"])

                        })
        # print(result)
        return result

    journey_list = journey_summary.apply(expand_journeys, axis=1)
    journey_list = pd.concat(journey_list.tolist())
    
    return(journey_list)

def map_one_hot(df, column_names, result_column_name):
    mapper = {} 
    for i, col_name in enumerate(column_names):
        for val in df[col_name].unique():
            mapper[str(val) + str(i)] = len(mapper)
         
    df_ext = df.copy()
    
    def one_hot(values):
        v = np.zeros( len(mapper) )
        for i, val in enumerate(values): 
            v[ mapper[str(val) + str(i)] ] = 1
        return v    
    
    df_ext[result_column_name] = df_ext[column_names].values.tolist()
    df_ext[result_column_name] = df_ext[result_column_name].map(one_hot)
    
    return df_ext
    

def UShape_attribution(df, n_campaigns=n_campaigns):
    
    def count_by_campaign(df, weights=None):
        counters = np.zeros(n_campaigns)
        # print(weights)
        for idx, campaign_one_hot in enumerate(df['campaigns'].values):
            campaign_id = np.argmax(campaign_one_hot)
            if weights is None:
              weight = 1
            else:
              weight = weights[idx]
            counters[campaign_id] = counters[campaign_id] + weight
        return counters
        
    campaign_impressions = count_by_campaign(df)
    #print(campaign_impressions)
    
    df_converted = df[df['conversion'] == 1]
    # print(df_converted.shape)
    first_mask = df_converted.groupby(['jid'])['timestamp_norm'].transform(lambda x: x == min(x))
    last_mask = df_converted.groupby(['jid'])['timestamp_norm'].transform(lambda x: x == max(x))
    journey_length = df_converted.groupby(['jid'])['timestamp_norm'].transform(len)
    other_mask = ~(first_mask | last_mask)

    #print(len(first_mask))
    attribution = np.array(first_mask,dtype="int")*0.4 + \
                  np.array(last_mask,dtype="int")*0.4 + \
                  np.multiply(np.array(other_mask,dtype="int") , 
                              np.divide(0.2, np.maximum(1e-10,np.array(journey_length,dtype="int")-2)))

    # Standardizing the values to sum up to one              
    attribution_df = pd.DataFrame({"jid":df_converted["jid"].values, "attribution": attribution})
    attribution = attribution_df.groupby(['jid'])['attribution'].transform(lambda x : x/x.sum())

    #print(attribution)
    #print(pd.Series(attribution).value_counts())
    campaign_conversions = count_by_campaign(df_converted, weights=attribution)
        
    return campaign_conversions / campaign_impressions
    

def plot_attribution(attribution, label, n_campaigns=n_campaigns):
    campaign_idx = range(0, n_campaigns)
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot(111)
    plt.bar( range(len(attribution[campaign_idx])), attribution[campaign_idx], label=label )
    plt.xlabel('Campaign ID')
    plt.ylabel('Return per impression')
    plt.legend(loc='upper left')
    plt.show()


def time_decay_attribution(df, n_campaigns=n_campaigns):
    
    def count_by_campaign(df, weights=None):
        counters = np.zeros(n_campaigns)
        # print(weights)
        for idx, campaign_one_hot in enumerate(df['campaigns'].values):
            campaign_id = np.argmax(campaign_one_hot)
            if weights is None:
              weight = 1
            else:
              weight = weights[idx]
            counters[campaign_id] = counters[campaign_id] + weight
        return counters
        
    campaign_impressions = count_by_campaign(df)
    #print(campaign_impressions)
    
    df_converted = df[df['conversion'] == 1]
    attribution_nr = df_converted.groupby(['jid'])['timestamp_norm'].transform(lambda x:x).values
    # print(attribution_nr)
    attribution_dr = df_converted.groupby(['jid'])['timestamp_norm'].transform(sum).values
    # print(attribution_dr)

    campaign_conversions = count_by_campaign(df_converted, weights=np.divide(attribution_nr,attribution_dr))
        
    return campaign_conversions / campaign_impressions



n_campaigns=20
demo_df = gen_journey_data()

demo_df = map_one_hot(demo_df, ['campaign'], 'campaigns').sort_values(by=['timestamp_norm'])
demo_df


usa = UShape_attribution(demo_df)
print(usa)
plot_attribution(usa, "U-Shaped attribution")
    
tda = time_decay_attribution(demo_df)
print(tda)
plot_attribution(tda, "Time decay attribution")