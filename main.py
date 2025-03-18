from src import train_model, MyNet


def main():
    model = MyNet()

    train_model(model)


if __name__ == "__main__":
    main()
