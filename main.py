import lightgbm as lgb
import pandas as pd
import shap
import streamlit as st
from streamlit_shap import st_shap


feature_desc_path = "./data/features_description.csv"
# train_data_path = "./data/hackathon_income_train.csv"
model_path = "./models/submission_a7.lgbm"
# shap_explainer_path = "./models/submition_a2.explainer"
cols_to_display = ["id", "gender", "adminarea", "age", "city_smart_name", "addref", "blacklist_flag",
                   "client_active_flag", "nonresident_flag", ""]

@st.cache_resource(max_entries=1, show_spinner="Загрузка модели")
def load_model(path):
    return lgb.Booster(model_file=path)


@st.cache_resource(max_entries=1, show_spinner="Загрузка модели интерпретации признаков")
def load_explainer(path):
    with open(path, "rb") as file:
        explainer = shap.TreeExplainer.load(file, instantiate=True)  # , model_loader=".save"
    return explainer


@st.cache_resource(show_spinner="Загрузка модели интерпретации признаков")
def calc_explainer(_model):
    explainer = shap.TreeExplainer(_model)
    return explainer


def predict_income(model, row) -> float:
    prediction = model.predict(row)
    return prediction


@st.cache_data(show_spinner="Вычисление влияния признаков")
def calc_shap_values(_explainer, row):
    return explainer(row)


def preprocess_data(row):
    row = row.drop(columns=["id", "dt"])
    row, *_ = convert_numeric_columns(row)
    num_list = []
    category_list = ["gender", "adminarea", "incomeValueCategory", "city_smart_name", "dp_ewb_last_employment_position",
                     "addrref", "dp_address_unique_regions", "blacklist_flag", "tz_msk_timedelta",
                     "vert_has_app_ru_tinkoff_investing", "client_active_flag", "nonresident_flag",
                     "vert_has_app_ru_vtb_invest", "vert_has_app_ru_cian_main", "vert_has_app_ru_raiffeisennews",
                     "period_last_act_ad", "accountsalary_out_flag"]
    for col in category_list:
        row[col] = row[col].astype('category')
    for col in num_list:
        row[col] = row[col].astype('float')
    return row


def convert_numeric_columns(df):
    num_columns = []
    cat_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            try:
                df[col] = df[col].astype(float)
            except Exception:
                cat_columns.append(col)
            else:
                num_columns.append(col)
        else:
            num_columns.append(col)
    return df, num_columns, cat_columns


#### HACK, HACK and HACK
def get_categorical_feature(model):
    import ast
    model_str_list = str(model.model_to_string()).split("\n")
    categorical_feature_str = list(filter(lambda x: 'categorical_feature' in x, model_str_list))[0]
    categorical_feature = ast.literal_eval("[" + categorical_feature_str.split()[1])
    return categorical_feature


@st.cache_data(show_spinner="Загрузка данных")
def load_dataframe(path, encoding="UTF-8"):
    return pd.read_csv(path, encoding=encoding, decimal=',', sep=';')


####
feature_desc = load_dataframe(feature_desc_path, encoding="windows-1251")
feature_desc_dict = feature_desc.set_index("признак")["описание"].to_dict()

model = load_model(model_path)
explainer = calc_explainer(model)
# train_data = load_dataframe(train_data_path)
# explainer = load_explainer(shap_explainer_path)

####

st.header("Сервис определения дохода клиента")
clients_file = st.file_uploader(label="Файл с данными по клиенту(ам)", type=["csv"])

if clients_file is not None:
    try:
        clients_df = pd.read_csv(clients_file, decimal=',', sep=';')
    except BaseException as exc:
        st.write("Не удалось обработать загруженный файл")
        raise exc

    st.write("Выберете клиента для просмотра подробного расчета")
    event = st.dataframe(clients_df, selection_mode="single-row", key="id", hide_index=True, on_select="rerun")
    if event.selection is not None and len(event.selection.rows):
        st.subheader("Информация по выбранному клиенту")
        selected_row_index = event.selection.rows[0]
        # st.write(selected_row_index)

        selected_row = clients_df[clients_df.index == selected_row_index]
        # Отображение выбранного клиента
        selected_row_display = selected_row.copy().T.reset_index()
        selected_row_display.columns = ["feature", "Значение"]
        selected_row_display["Название параметра"] = selected_row_display["feature"].map(feature_desc_dict)
        st.dataframe(selected_row_display[selected_row_display["feature"].isin(cols_to_display)][
                         ["Название параметра", "Значение"]], hide_index=True)
        # Вывод предсказанного значения
        # f = model.()
        # cf = ",".join(map(lambda i: f'"{f[i]}"', get_categorical_feature(model)))
        # for i in get_categorical_feature(model):
        # st.write(cf)
        selected_row = preprocess_data(selected_row)
        with st.spinner("Вычисление дохода"):
            predicted_income = float(predict_income(model, selected_row))
            st.write(f"Предсказанное значение дохода: **{predicted_income:2,.2f}**".replace(",", " "))

            # Вывод анализа признаков
            #     selected_row_shap = selected_row.copy().rename(mapper=feature_desc_dict, axis=1)
            #     st.dataframe(selected_row)
            shap_values = calc_shap_values(explainer, selected_row)

        st.subheader("Анализ признаков клиента")
        st_shap(shap.force_plot(shap_values, features=selected_row), height=200,
                width=1000)
        st_shap(shap.plots.waterfall(shap_values[0]))
        st_shap(shap.plots.beeswarm(shap_values))
