from uuid import UUID

import ipywidgets as widgets
from IPython.display import display
from langsmith import Client


def display_feedback_buttons(run_id: UUID) -> None:
    # GoodボタンとBadボタンを準備
    good_button = widgets.Button(
        description="Good",
        button_style="success",
        icon="thumbs-up",
    )
    bad_button = widgets.Button(
        description="Bad",
        button_style="danger",
        icon="thumbs-down",
    )

    # クリックされた際に実行される関数を定義
    def on_button_clicked(button: widgets.Button) -> None:
        if button == good_button:
            score = 1
        elif button == bad_button:
            score = 0
        else:
            raise ValueError(f"Unknown button: {button}")
        client = Client()
        # LangSmithにフィードバックを保存
        client.create_feedback(run_id=run_id, key="thumbs", score=score)
        print("フィードバックを送信しました")

    # ボタンがクリックされた際にon_button_clicked関数を実行
    good_button.on_click(on_button_clicked)
    bad_button.on_click(on_button_clicked)

    # ボタンを表示
    display(good_button, bad_button)


from langchain_core.tracers.context import collect_runs

# LangSmithのトレースのID(Run ID)を取得するため、collect_runs関数を使用
with collect_runs() as run_cb:
    output = chain.invoke("LangChainの概要を教えて")
    print(output["answer"])
    run_id = run_cb.traced_runs[0].id

display_feedback_buttons(run_id)
