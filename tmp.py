##In trainer.py, Using with Wiki

# if not self.wiki:
#     ticker_df = bertopic_fun.ticker2concept(
#         topic_model, self.concept_dict, self.total_ticker, BERTopic_concept, results_df, self.concept_n, self.threshold_sim)

#     data_utils.save_file(concept_path, 'result_ticker', ticker_df)

# else:
#     pass
    # TODO: Need more check
    # keyword = input("Enter Keyword: ")
    # keyword = "_".join(keyword.split(' '))
    # with open('./{}_filter.json'.format(keyword), 'r') as f:
    #     string = f.read()
    #     sub_cat = json.loads(json.loads(string))

    # del sub_cat[keyword]
    # sub_concept = list(sub_cat.keys())
    # results_sub = results_df[results_df.concept == keyword]
    # sub_tickers = list(results_sub.origin +
    #                    results_sub.common + results_sub.bertopic)[0]

    # results_df2 = bertopic_wiki.subconcept2ticker(
    #     topic_model, keyword, sub_concept, sub_tickers, classifier, BERTopic_ticker, args.concept_N, args.threshold_sim, 2)

    # results_df2.to_csv(f"{keyword}_concept.csv")