def first_touch_attribution(df):
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
    print(campaign_impression)
    df_converted = df[df['conversion']==1]
    idx = df_converted.groupby(['jid'])['timestamp_norm'].transform(max) == df_converted['timestamp_norm']
    campaign_converted = count_by_campaign(df_converted[idx])
    return campaign_converted/campaign_impression