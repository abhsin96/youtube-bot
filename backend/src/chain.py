from langchain_core.runnables import RunnableLambda


def build_hello_chain():
    return RunnableLambda(lambda x: f"Hello, {x}!")


if __name__ == "__main__":
    chain = build_hello_chain()
    result = chain.invoke("world")
    print(result)
