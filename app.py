import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def main():
    st.title("機械学習の学習結果可視化アプリ")

    # データセットのアップロード
    dataset = st.sidebar.selectbox("データセットを選択", ("アイリス", "ワイン"))
    st.write(f"データセット：{dataset}")

    # モデル選択
    classifier = st.sidebar.selectbox(
        "モデルを選択", ("ロジスティック回帰", "サポートベクターマシン", "ランダムフォレスト"))

    # ハイパーパラメータの設定
    if classifier == "ロジスティック回帰":
        st.sidebar.subheader("ハイパーパラメータ")
        C = st.sidebar.number_input(
            "正則化パラメータ C", 0.01, 10.0, step=0.01, value=0.01)
        model = LogisticRegression(C=C)
    elif classifier == "サポートベクターマシン":
        st.sidebar.subheader("ハイパーパラメータ")
        C = st.sidebar.number_input(
            "コストパラメータ C", 0.01, 10.0, step=0.01, value=1.0)
        kernel = st.sidebar.radio("カーネル", ("linear", "rbf"))
        model = SVC(C=C, kernel=kernel)
    else:
        st.sidebar.subheader("ハイパーパラメータ")
        n_estimators = st.sidebar.number_input(
            "決定木の数", 10, 200, step=10, value=10)
        max_depth = st.sidebar.slider("最大深さ", 1, 10, value=5)
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth)

    # データセットの読み込み
    if dataset == "アイリス":
        # Irisデータセット
        from sklearn.datasets import load_iris
        data = load_iris()
    else:
        # Wineデータセット
        from sklearn.datasets import load_wine
        data = load_wine()

    X = data.data
    y = data.target

    # データの確認
    if st.button("データの確認"):
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target_names[data.target]
        st.dataframe(df)

    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # モデルの学習の開始
    if st.button("学習開始"):
        # モデルの学習
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 可視化
        st.subheader("学習結果")
        st.write("精度: ", model.score(X_test, y_test))

        st.subheader("混同行列")
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True,
                    cmap='coolwarm', ax=ax, cbar=False, fmt='d', linewidths=1)
        st.pyplot(fig)

        st.subheader("分類レポート")
        st.write(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
