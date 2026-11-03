import matplotlib.pyplot as plt


def plot_class_distribution(class_distribution):

    class_distribution.plot(
        x="Class",
        y="Count",
        kind="pie",
        legend=False,
        color=["blue", "green"]
    )

    plt.ylabel("Number of observations")
    plt.title("Class of distribution")
    plt.xticks(rotation=0)

    plt.show()