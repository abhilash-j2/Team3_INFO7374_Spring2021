def UShape_attribution(df, n_campaigns=n_campaigns):
    
    def count_by_campaign(df, weights=None):
        counters = np.zeros(n_campaigns)
        print(weights)
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

    print(attribution)
    print(pd.Series(attribution).value_counts())
    campaign_conversions = count_by_campaign(df_converted, weights=attribution)
        
    return campaign_conversions / campaign_impressions
    