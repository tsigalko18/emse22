import csv

from statistical_tests.wilcoxon_test import *


def compute_statistics(online_steering_angles, offline_steering_angles):
    simulation1 = np.asarray(online_steering_angles, dtype=float)
    simulation2 = np.asarray(offline_steering_angles, dtype=float)

    try:
        assert len(simulation1) == len(simulation2)
    except AssertionError:
        print(len(simulation1))
        print(len(simulation2))

    w_statistic, pvalue = wilcoxon(simulation1, simulation2)
    cohensd = cohend(simulation1, simulation2)

    pow = run_power_analysis_two_sets(simulation1, simulation2)

    return pvalue, cohensd[0], pow


def read_rq1_data(scenario):
    actual = []
    autumn = []
    chauffeur = []

    if scenario == 30:
        return None

    d = "data/replicated-paper-data-rq0/with-ground-truth/" + str(scenario)

    f = open(d + '/' + 'Actual.txt')
    for y in f.read().split('\n'):
        try:
            actual.append(float(y))
        except ValueError as exception:
            actual.append(float(0.0))

    assert len(actual) > 0

    f = open(d + '/' + 'Autumn.txt')
    for y in f.read().split('\n'):
        try:
            autumn.append(float(y))
        except ValueError as exception:
            autumn.append(float(0.0))

    assert len(autumn) > 0

    f = open(d + '/' + 'Chauffeur.txt')
    for y in f.read().split('\n'):
        try:
            chauffeur.append(float(y))
        except ValueError:
            chauffeur.append(float(0.0))

    assert len(chauffeur) > 0

    return actual, autumn, chauffeur


def compute_stat_significance(actual, autumn, chauffeur):
    print(compute_statistics(actual, autumn))
    print(compute_statistics(actual, chauffeur))
    print("\n")


def compute_mae_score(actual, model):
    mean_actual = np.mean(actual)
    mean_model = np.mean(model)

    return abs(mean_actual - mean_model)


def is_mae_scores_below_threshold(actual, model, t):
    mean_actual = np.mean(actual)
    mean_model = np.mean(model)

    if abs(mean_actual - mean_model) <= t:
        return True
    else:
        return False


if __name__ == '__main__':
    # STAT. SIGNIFICANCE OF RQ1
    # for scenario in range(1, 32):
    #     actual, autumn, chauffeur = read_briands_rq1_data(scenario)
    #     # compute_mae_scores(actual, autumn, chauffeur)
    #     compute_stat_significance(actual, autumn, chauffeur)

    # MAE ANALYSIS FOR DIFFERENT THRESHOLDS OF RQ1
    all_mae_scores = []
    all_actual = []

    header = ["sim", "MAE", "MAE (deg)", "p-value", "eff. size", "pow", "sim", "MAE", "MAE (deg)", "p-value",
              "eff. size", "pow"]

    with open('results/rq0-results-replicated-paper.csv', 'w', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

    f.close()

    for scenario in range(1, 33):

        try:
            actual, autumn, chauffeur = read_rq1_data(scenario)
            all_actual.append(actual)
        except TypeError:
            continue

        mae_scores_for_scenario = []
        row = []

        for model in [autumn, chauffeur]:
            mae_scores = []
            for t in [0.1]:
                mae = compute_mae_score(actual, model)
                mae_in_deg = mae * 25
                mae_scores.append(mae)

            mae_scores_for_scenario.append(mae_scores)
            pvalue, effsize, pow = compute_statistics(actual, model)
            row.append(scenario)
            row.append(mae)
            row.append(mae_in_deg)
            row.append(pvalue)
            row.append(effsize)
            row.append(pow)

        with open('results/rq0-results-replicated-paper.csv', 'a', newline='', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    f.close()
