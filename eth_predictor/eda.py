import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import dataframe_image as dfi


def plot_year_target(eth_data):
    """

    Plotting the closing price for each day throughout the year to determine the range of years our neural network should
    focus on to learn the trend. Results have shown that the ETH-USD price have risen dramatically after 2020 ; hence, we 
    will only focus between 2020-11-01 to the current date to prevent the neural network from learning the more stable 
    trend in previous years.

    Args:
        eth_data (Pandas Dataframe): A Dataframe containing the target and features for each date.
    """

    # Iterating through each year and plotting the closing price for each one.
    for year in ['2018', '2019', '2020', '2021', '2022', '2023']:
        start_year = year + "-01-01"
        end_year = year + "-12-31"
        # Indexing the ETH-USD information for all the days within a single year
        year_data = eth_data[(eth_data['Date'] >= start_year)
                             & (eth_data['Date'] <= end_year)]
        year_target = year_data['Target']
        print(year_target)
        plt.plot(list(year_target.index), year_target, label=year)

    plt.xlabel("Date")
    plt.ylabel("ETH-USD Price")
    plt.title("ETH-USD Closing Price Change From 2018-2023")
    plt.legend()
    plt.show()


def plot_year_summary(eth_data):
    """

    Plotting a summary statistics of each feature for each year between 2021-2023

    Args:
        eth_data (Pandas Dataframe): A Dataframe containing the target and features for each date.
    """

    # Iterating through each year and plotting the closing price for each one.
    for year in ['2018', '2019', '2020', '2021', '2022', '2023']:
        start_year = year + "-01-01"
        end_year = year + "-12-31"
        # Indexing the ETH-USD information for all the days within a single year
        year_data = eth_data[(eth_data['Date'] >= start_year)
                             & (eth_data['Date'] <= end_year)]
        feature_data = year_data.drop(['Date'], axis=1)
        # Creating a Dataframe to store the summary statistics for each feature
        summary_stats = pd.DataFrame(
            feature_data.columns, columns=['Features'])

        # Iterating through each feature in the yearly Dataframe
        for i, feature in enumerate(feature_data.columns):
            feature_col = feature_data[feature]
            # Computing the summary statistics using all values from a feature within a single year
            summary_stats.loc[i, "Min"] = np.amin(feature_col)
            summary_stats.loc[i, "Mean"] = np.mean(feature_col)
            summary_stats.loc[i, "Max"] = np.amax(feature_col)
            summary_stats.loc[i, "Std"] = np.std(feature_col)
            summary_stats.loc[i, "25%"] = np.percentile(feature_col, 25)
            summary_stats.loc[i, "50%"] = np.percentile(feature_col, 50)
            summary_stats.loc[i, "75%"] = np.percentile(feature_col, 75)

        dfi.export(summary_stats, "eda_result/" + year + "/summary_stats.png")


def plot_summary(eth_data):
    """

    Plotting a summary statistics of each feature throughout the years between 2020 and 2023


    Args:
        eth_data (Pandas Dataframe): A Dataframe containing the target and features for each date.
    """
    feature_data = eth_data.drop(['Date'], axis=1)
    # Creating a Dataframe to store the summary statistics for each feature
    summary_stats = pd.DataFrame(
        feature_data.columns, columns=['Features'])

    # Iterating through each feature between the year 2020 and 2023
    for i, feature in enumerate(feature_data.columns):
        feature_col = feature_data[feature]
        # Computing the summary statistics using all values from a feature between 2020 and 2023
        summary_stats.loc[i, "Min"] = np.amin(feature_col)
        summary_stats.loc[i, "Mean"] = np.mean(feature_col)
        summary_stats.loc[i, "Max"] = np.amax(feature_col)
        summary_stats.loc[i, "Std"] = np.std(feature_col)
        summary_stats.loc[i, "25%"] = np.percentile(feature_col, 25)
        summary_stats.loc[i, "50%"] = np.percentile(feature_col, 50)
        summary_stats.loc[i, "75%"] = np.percentile(feature_col, 75)

    dfi.export(summary_stats, "eda_result/overall/summary_stats.png")


def plot_feature_distribution(eth_data):
    """

    Plot the distribution and box plot of each feature throughout the years between 2020 and 2023. These 
    plots allow us to determine how to preprocess the dataset. Results have shown that the distribution of most 
    features are not normal so we should use normalisation to scale the dataset.

    Args:
        eth_data (Pandas Dataframe): A Dataframe containing the target and features for each date.
    """
    feature_data = eth_data.drop(['Date'], axis=1)

    # Iterating through each feature between the year 2020 and 2023
    for feature in feature_data.columns:
        feature_col = feature_data[feature]
        # Plotting the histogram of a feature's value for all the dates between 2020 and 2023
        plt.hist(feature_col, bins=20, edgecolor='black')
        plt.xlabel(feature)
        plt.ylabel("Value")
        plt.title("Distribution Plot of " + feature)

        # Saving the histogram plot of the feature into a picture
        plt.savefig("eda_result/overall/" + feature + "_distribution.png")
        plt.close()

        # Plotting the box plot of a feature's value for all the dates between 2020 and 2023
        plt.boxplot(feature_col, vert=False)
        plt.xlabel("Value")
        plt.yticks([1], [feature])
        plt.title("Box Plot of " + feature)

        # Saving the box plot of the feature into a picture
        plt.savefig("eda_result/overall/" + feature + "_box.png")
        plt.close()


def plot_year_trend(eth_data):
    """

    Plot to show how the value of each feature varies with the target variable in each year between 2020 and 2023

    Args:
        eth_data (Pandas Dataframe): A Dataframe containing the target and features for each date.
    """
    for year in ['2018', '2019', '2020', '2021', '2022', '2023']:
        start_year = year + "-01-01"
        end_year = year + "-12-31"
        # Indexing the ETH-USD information for all the days within a single year
        year_data = eth_data[(eth_data['Date'] >= start_year)
                             & (eth_data['Date'] <= end_year)]
        year_target = year_data['Target']
        feature_data = year_data.drop(['Date', 'Target'], axis=1)

        # Iterating through each feature between the year 2020 and 2023
        for feature in feature_data.columns:
            # Obtaining the feature value for each day in a year
            feature_col = feature_data[feature]
            # Plotting the target label and feature value on the same plot
            plt.plot(feature_col, label=feature)
            plt.plot(year_target, label="Target")

            plt.xlabel("Date")
            plt.ylabel("ETH-USD Price")
            plt.title(feature + " Feature Trend in " + year)
            plt.legend()

            # Saving the feature-target correlation plot into a picture
            plt.savefig("eda_result/" + year + "/" + feature + ".png")
            plt.close()


eth_data = pd.read_csv("../coin_data/eth_data.csv")
plot_year_target(eth_data)
plot_year_summary(eth_data)
plot_summary(eth_data)
plot_year_trend(eth_data)
plot_feature_distribution(eth_data)
