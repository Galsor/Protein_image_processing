from sklearn.model_selection import train_test_split

#TODO ne conserver que les modèles de clustering
CLASSIFIERS =((RidgeClassifier(tol=1e-5, solver="lsqr"), "Ridge Classifier"),
            #(Perceptron(max_iter = 25000), "Perceptron"),
            (LogisticRegression(n_jobs=1, C=1e6), "Logisitic Regression"),
            (MLPClassifier(hidden_layer_sizes=(75,), max_iter=25000, alpha=0.00001), "MLPC_V1"),
            (MLPClassifier(hidden_layer_sizes= (80,), max_iter=30000, alpha = 1e-5), "MLPC_V2"),
            (MLPClassifier(hidden_layer_sizes=(80,), max_iter= 30000, alpha=1e-7), "MLPC_V3"),
            #(PassiveAggressiveClassifier(), "Passive-Aggressive"),
            #(KNeighborsClassifier(n_neighbors=50), "kNN"),
            (RandomForestClassifier(n_estimators=100), "Random forest"),
            #(RandomForestRegressor(n_estimators=100), "Random Forest Regressor"),
            #(SGDClassifier(alpha=.00001,penalty="l2"), "SGDC l2"),
            #(NearestCentroid(), "NearestCentroid"),
            (BernoulliNB(alpha=.01), "BenouilliNB"),
            (LinearSVC(loss='l2', penalty="l2", dual=False, tol=1e-5), "Linear SVC"),
            (GaussianNB(),"GaussianNB"))

# TODO Ajouter l'accès aux trains data
EMBRYOS = {1: (77, 24221), 7: (82, 23002), 8: (71, 15262), 10: (92, 23074)}


class MultimodelPredictor:
    def __init__(self, path=data_path):
        # Get and prep data for training
        self.init = True
        #get Training data
        raw_df, self.df = get_data(path)
        # Set pipeline for features extraction
        self.pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer())])
        df_vect = self.prep_data(self.df)
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_data(df_vect)

        # Instanciate and embryos models

        self.classifiers = self.get_best_classifiers(0.70)
        self.init = False

    def get_best_classifiers(self, ratio):
        results = []
        for clf, name in CLASSIFIERS:
            try:
                clf.fit(self.x_train, self.y_train)
                pred = clf.predict(self.x_test)
                if name in ["Random Forest Regressor", "MPLR"]:
                    pred = pred.round()
                score = metrics.accuracy_score(self.y_test, pred)
                logging.info(name + " score: " + str(score * 100) + " %")
                results.append((clf, name, score))
            except Exception as e:
                logging.warning("Fail in training " + name)
                logging.warning(repr(e))
                traceback.print_tb(e.__traceback__)
                t.sleep(2)

        logging.info("Models tested : ")
        logging.info(len(results))
        classifiers = [result for result in results if result[2] > ratio]
        logging.info("Models retained : ")
        logging.info(len(classifiers))
        return classifiers

    def split_data(self, df):
        #TODO définir y et l'enlever du dataset
        y = df["Type"]
        x = df.drop("Type", axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)
        return x_train, x_test, y_train, y_test

    def prep_data(self, data):
        # TODO Prepare data before feature extaction
        return df_vect

    def predict_all(self, df):
        preds = pd.DataFrame()
        for clf, name, score in self.classifiers:
            try:
                preds = pd.concat([preds, pd.DataFrame(clf.predict(df), columns=[name])], axis=1)
            except Exception as e:
                logging.warning("=" * 80)
                logging.warning("Fail in predicting with " + name)
                logging.warning("_" * 80)
                logging.warning(repr(e))
        logging.info("Predictions")
        logging.info("=" * 80)
        logging.info(preds)

        #TODO compute classification score
        scores = compute_scores(preds)
        #TODO Retains the best predictions
        df_res = self.method_max(scores)

        return df_res

    def labelise(self, input, transcript=True, print_results=True):
        if isinstance(input, str):
            try:
                raw_df, df_x = get_data(input)
            except Exception as e:
                raise e
        elif isinstance(input, pd.DataFrame):
            logging.info(input.columns)
            df_x = input.reset_index().drop('index', axis=1)
            raw_df = df_x

        else:
            raise ValueError("Input must be String (path) or pd.DataFrame")

        if any(col in df_x.columns for col in ('Libellé', 'LibellÃ©')):
            df_vect = self.prep_data(data=df_x)
        else:
            df_vect = df_x

        if 'Type' in df_vect.columns:
            df_vect = df_vect.drop("Type", axis=1)

        df_pred = self.predict_all(df_vect)

        if print_results:
            self.print_results(df_pred)

        if transcript:
            df_pred = self.transcript_results(df_pred, raw_df)
        return df_pred


    def method_max(self, scores):
        pred = scores.idxmax(axis=1)
        total_preds = scores.iloc[0].sum()
        confusion_scores = []
        for row in scores.index.values:
            vals = scores.iloc[row]
            amount_of_pred = vals.loc[pred[row]]
            confusion_scores.append(amount_of_pred / total_preds)
        confusion_scores = pd.Series(confusion_scores)

        pred = pd.concat([pred, confusion_scores], axis=1)
        pred.columns = ["Predictions", "Confusion score"]

        logging.info("\n Result of Max Method")
        logging.info("=" * 80)
        logging.info(pred)
        return pred

    def print_results(self, df_res):
        try:
            print('Results on the test set:')
            print("Accuracy : " + str(accuracy_score(self.y_test, df_res["Predictions"])))
            print(classification_report(self.y_test, df_res["Predictions"]))
        except Exception as e:
            logging.warning("Impossible to print results of classification")
            logging.warning(repr(e))