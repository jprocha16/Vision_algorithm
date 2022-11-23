import numpy as np
import cv2
import math
import time
import pandas as pd
from scipy.interpolate import interp1d


def compute_ground_sampling_distance(focal_distance, sensor_width, flight_height, image_width,
                                     image_height):
    gsd = round((sensor_width * flight_height * 100) / (focal_distance * image_width), 3)     # cm/pixel
    return gsd


#def compute_max_ring_size(gsd, outer_ring_radius):
    #max_radius = int(round((outer_ring_radius / gsd) + 10, 0))     # size in pixels (real = 40)
    #return max_radius


#def compute_min_ring_size(gsd, inner_ring_radius):
    #min_radius = int(round((inner_ring_radius / gsd) - 10, 0))     # size in pixels (real = 1.3)
    #return min_radius


def get_parameters(flight_height, file_name):
    df = pd.read_excel(file_name)

    height = df["flight_height"]
    min_radius = df["min_radius"]
    max_radius = df["max_radius"]
    step = df["step"]
    center_error = df["center_error"]
    radius_error = df["radius_error"]
    concentric_error = df["concentric_error"]

    f_min_radius = interp1d(height, min_radius)
    f_max_radius = interp1d(height, max_radius)
    f_step = interp1d(height, step)
    f_center_error = interp1d(height, center_error)
    f_radius_error = interp1d(height, radius_error)
    f_concentric_error = interp1d(height, concentric_error)

    min_radius = f_min_radius(flight_height)
    max_radius = f_max_radius(flight_height)
    step = f_step(flight_height)
    center_error = f_center_error(flight_height)
    radius_error = f_radius_error(flight_height)
    concentric_error = f_concentric_error(flight_height)

    return {
        'min_radius': min_radius,
        'max_radius': max_radius,
        'step': step,
        'center_error': center_error,
        'radius_error': radius_error,
        'concentric_error': concentric_error
    }


def detect_circles(im: np.ndarray, step, minR: int=0, maxR: int=0):
    circles = []
    minR = max(minR, 0)
    #step = 20
    for maxR in range(minR+step, maxR, step):
        new_circles = cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, 1, 100,
                                   param1=100, param2=100, minRadius=minR, maxRadius=maxR)
        minR += step
        if new_circles is not None:
            circles.extend(new_circles[0].tolist())
    circles = list(map(tuple, circles))
    return sorted(circles, key=lambda circle: circle[2], reverse=True)


def draw_circles(circles, im):
    for c in circles:
        c = list(map(int, c))
        x,y,r = c
        # Draw the circle in the output image
        cv2.circle(im, (x, y), r, (0, 255, 0), 1)
        # Draw a rectangle(center) in the output image
        cv2.rectangle(im, (x - 2, y - 2), (x + 2, y + 2), (0, 255, 0), -1)


def get_unique_circles(circles_input, center_error, radius_error, strategy):
    '''
    :param circles_input: list of sorted circles, bigger radii first
    :param strategy: way to obtain the equivalent circle
    :return: list of unique circles, according to the defined strategy, sorted, bigger radii first
    '''

    unique = []
    while len(circles_input) > 0:
        c1 = circles_input[0]
        similar = [c1]

        # search for similar circles to c1
        for c2 in circles_input:
            if is_similar(c1, c2, center_error=center_error, radius_error=radius_error):
                similar.append(c2)

        # remove similar circles from original list
        for c in similar:
            if c in circles_input:
                circles_input.remove(c)

        # compute equivalent circle
        final_circle = combined_circles(similar, strategy=strategy)
        unique.append(final_circle)
    return unique


def is_similar(c1, c2, center_error, radius_error):
    '''
    :param center_error: error allowed between two circles centers to consider them the same circle
    :param radius_error: error allowed between two circles radii to consider them the same circle
    :return: list of similar circles, given the admitted errors, sorted, bigger radii first
    '''
    x1, y1, r1 = c1
    x2, y2, r2 = c2

    if math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) < center_error and abs(r2 - r1) < radius_error:
        return True
    else:
        return False


def combined_circles(similar, strategy):
    '''
    :param strategy:mean or biggest
    :param similar: list of similar circles
    :return: a single circle for each list, computed accorded the defined strategy
    '''
    if strategy == 'mean':
        x_avg = 0
        y_avg = 0
        r_avg = 0
        for (x, y, r) in similar:
            x_avg += x
            y_avg += y
            r_avg += r

        x_avg = round(x_avg / len(similar), 2)
        y_avg = round(y_avg / len(similar), 2)
        r_avg = round(r_avg / len(similar), 2)
        return ((x_avg, y_avg, r_avg))

    elif strategy == 'biggest':
        return similar[0]


def get_concentric_circles(circles_input, concentric_error):
    '''
    :param concentric_error: error allowed in pixels to consider two circles concentrical
    :param circles_input: list of circles, each circle a tuple with center and radius
    :return: list of lists, each list has a set of concentric circles
    '''
    concent = []
    while len(circles_input) > 0:
        circle = circles_input.pop(0)
        group = [circle]
        for (x1, y1, r1) in circles_input:
            (x2, y2, r2) = group[-1]
            if math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) < concentric_error:
                group.append((x1, y1, r1))
        for circle in group:
            if circle in circles_input:
                circles_input.remove(circle)
        concent.append(group)
    return concent


def get_radii_ratios(circles_list):
    ratios = []
    if len(circles_list) < 2:
        return ratios
    for circle_counter in range(0, len(circles_list) - 1):
        (x1, y1, r1) = circles_list[circle_counter]
        (x2, y2, r2) = circles_list[circle_counter + 1]
        ratios.append(round(r2 / r1, 3))
    return ratios


def compare_ratio_sequence(ratios_list):
    signature = [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.5] #NÃO DEVERIA ESTAR DEFINIDA FORA E NÃO NA FUNÇÃO?
    list_match = []
    for obs in range(0, len(ratios_list)):
        for real in range(0, len(signature)):
            upper_lim = signature[real] + 0.02
            lower_lim = signature[real] - 0.02
            if ratios_list[obs] <= upper_lim and ratios_list[obs] >= lower_lim:
                list_match.append([obs, real])
    return list_match


def get_longest_list_of_matches(sequence_match):  # CRITÉRIO 1
    # tamanho da maior familia
    # seq[0] -> indice da familia no tuple
    # seq[1] -> familia de racios
    # len(seq[1]) -> tamanho da familia de racios
    max_size = max(map(lambda seq: len(seq[1]), sequence_match))

    # lista das familias com o maior tamanho
    def family_is_longest(seq):
        return len(seq[1]) == max_size
    longest_families = list(filter(family_is_longest, sequence_match))

    return longest_families


def get_outer_match_list(longest_match_list):  # CRITÉRIO 2
    def value_to_compare_in_seq(seq: tuple):
        # seq[0] -> indice da familia no tuple
        # seq[1] -> familia de racios
        # seq[1][0] -> primeiro racio da familia de racios
        # seq[1][0][1] -> match do primeiro racio da familia de racios
        return seq[1][0][1]

    return min(longest_match_list, key=value_to_compare_in_seq)


def main():
    image = cv2.imread("/media/sf_Shared_folder_Ubuntu/test.jpg", 0)
    output = cv2.imread("/media/sf_Shared_folder_Ubuntu/test.jpg", 1)

    cv2.namedWindow('original image', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('original image', output)
    cv2.resizeWindow('original image', 700, 700)
    # cv2.waitKey()

    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    cv2.namedWindow('blurred image', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('blurred image', blurred)
    cv2.resizeWindow('blurred image', 700, 700)
    # cv2.waitKey()

    flight_height = 2
    print('flight_height = ', flight_height)
    gsd = compute_ground_sampling_distance(focal_distance=18, sensor_width=22.3, flight_height=flight_height,
                                           image_width=5184, image_height=3456)
    print("gsd = ", gsd)


    params = get_parameters(flight_height, "/media/sf_Shared_folder_Ubuntu/Detection_results_v2.xlsx")
    params['min_radius']
    params['max_radius']
    params['step']
    params['center_error']
    params['radius_error']
    params['concentric_error']

    print('min_radius = ', params['min_radius'])
    print('max_radius = ', params['max_radius'])
    print('step = ', params['step'])
    print('center_error = ', params['center_error'])
    print('radius_error = ', params['radius_error'])
    print('concentric_error = ', params['concentric_error'])

    t_start = time.time()
    sorted_circles = detect_circles(blurred, maxR=int(params['max_radius']), minR=int(params['min_radius']),
                                    step=int(params['step']))
    print(f'circle detection in {round(time.time() - t_start, 1)} seconds')

    draw_circles(sorted_circles, output)
    cv2.namedWindow('circle detection', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('circle detection', output)
    cv2.resizeWindow('circle detection', 700, 700)
    cv2.waitKey()

    print(sorted_circles)

    unique_circles = get_unique_circles(sorted_circles, center_error=params['center_error'],
                                        radius_error=params['radius_error'], strategy='mean')
    print(unique_circles)

    concentric_circles = get_concentric_circles(unique_circles, concentric_error=params['concentric_error'])
    print(concentric_circles)

    radii_ratios = list(map(get_radii_ratios, concentric_circles))
    print(radii_ratios)

    sequence_match = list(map(compare_ratio_sequence, radii_ratios))
    print(sequence_match)

    sequence_match = [(i, seq) for i,seq in enumerate(sequence_match)]

    longest_match_list = get_longest_list_of_matches(sequence_match)
    print(longest_match_list)

    if len(longest_match_list) > 1:
        outer_match_list = get_outer_match_list(longest_match_list)
        print(outer_match_list)
        final_match = outer_match_list
    else:
        final_match = longest_match_list[0]
    print(concentric_circles[final_match[0]])

if __name__ == '__main__':
    main()