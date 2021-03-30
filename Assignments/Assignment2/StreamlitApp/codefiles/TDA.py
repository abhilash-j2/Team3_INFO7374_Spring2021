def time_decay_attribution(df, n_campaigns=n_campaigns):
    
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
    attribution_nr = df_converted.groupby(['jid'])['timestamp_norm'].transform(lambda x:x).values
    print(attribution_nr)
    attribution_dr = df_converted.groupby(['jid'])['timestamp_norm'].transform(sum).values
    print(attribution_dr)

    campaign_conversions = count_by_campaign(df_converted, weights=np.divide(attribution_nr,attribution_dr))
        
    return campaign_conversions / campaign_impressions
