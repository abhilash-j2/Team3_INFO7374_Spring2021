def linear_attribution(df):
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
    campaign_impression = count_by_campaign(df)
    df_converted = df[df['conversion']==1]
    journey_length = df_converted.groupby('jid')['timestamp_norm'].transform(len).values 
    weight = 1/journey_length
    campaign_conversions = count_by_campaign(df_converted, weights=weight)
    return campaign_conversions / campaign_impression