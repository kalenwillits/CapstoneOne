def model(random_state):
    table_columns = ["Model", "ExplainedVariance", "MeanAbsoluteError"]
    table = {}
    for column_label in table_columns:
        table[column_label] = []

    dfo = df.select_dtypes(include=['object']) # select object type columns
    df = pd.concat([df.drop(dfo, axis=1), pd.get_dummies(dfo)], axis=1)

    X = df.drop(['AdultWeekend'], axis=1)
    y = df.AdultWeekend
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    y = y
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=random_state)

    lm = linear_model.LinearRegression()
    model = lm.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    exp_y = explained_variance_score(y_test, y_pred)
    mean_abs_y = sklearn.metrics.mean_absolute_error(y_test, y_pred)


    table['Model'].append('2')
    table['ExplainedVariance'].append(exp_y)
    table['MeanAbsoluteError'].append(mean_abs_y)

    return table
