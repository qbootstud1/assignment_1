import math
from math import floor
from typing import Tuple, List, Union, Any
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt


def print_chart(canvas: NDArray) -> None:
    """
    Print a multidimensional array as a joint string
    :param canvas: Array of strings to print
    :return: Joint array of strings
    """
    for row in range(len(canvas)):
        print("".join(canvas[row]))


def gen_canvas(height_size: int, width_size: int, grid_opt: bool = False) -> NDArray:
    """
    Generates an empty canvas as a 2D array
    :param height_size: Height of the canvas
    :param width_size: Width of the canvas
    :param grid_opt: If True, a grid is displayed
    :return: 2D array
    """
    assert all(isinstance(size, int) for size in (height_size, width_size))

    canvas = np.empty((height_size, width_size), dtype="str")
    canvas[:] = ' '
    canvas[::height_size - 1] = '+'
    canvas[::height_size - 1] = '+'
    canvas[::height_size - 1, 1:-1] = '-'
    canvas[1:-1, ::width_size - 1] = '|'
    if grid_opt:
        canvas[1:-1, 1:-1] = '·'

    return canvas


def add_xaxis_canvas(x_vec: Union[List[Union[float]], NDArray], width_size: int, legend: str = ' ') -> NDArray:
    """
    Generates a 2D array for the x axis of a plot or canvas
    :param x_vec: Vector with the x data points
    :param width_size: Width of the plot or canvas
    :param legend: Legend of the plot to be displayed within the x axis
    :return: 2D Array
    """
    assert len(legend) <= width_size

    middle_array = np.empty((1, width_size - 2), dtype="str")
    middle_array[:] = ' '
    lateral_spaces = floor((width_size - 2 - len(legend)) / 2)
    for elem in range(len(legend)):
        middle_array[0][lateral_spaces + elem] = list(legend)[elem]
    min_array = np.empty((1, len(str(floor(x_vec[0])))), dtype="str")
    for elem in range(len(str(round(x_vec[0])))):
        min_array[:, elem] = list(str(round(x_vec[0])))[elem]
    max_array = np.empty((1, len(str(round(x_vec[-1])))), dtype="str")
    for elem in range(len(str(round(x_vec[-1])))):
        max_array[:, elem] = list(str(round(x_vec[-1])))[elem]
    array = np.hstack((min_array, middle_array,  max_array))

    return array


def add_yaxis_canvas(y_vec: Union[List[Union[float]], NDArray], height_size: int) -> NDArray:
    """
    Generates a 2D array for the y axis of a plot or canvas
    :param y_vec: Vector with the y data points
    :param height_size: Height of the plot or canvas
    :return: 2D Array
    """
    middle_array = np.empty((height_size - 2, max(len(str(round(min(y_vec)))), len(str(round(max(y_vec)))))),
                            dtype="str")
    middle_array[:] = ' '
    min_array = np.empty((1, max(len(str(round(min(y_vec)))), len(str(round(max(y_vec)))))), dtype="str")
    min_array[:] = ' '
    for elem in range(len(str(round(min(y_vec))))):
        min_array[:, -1 - elem] = list(str(round(min(y_vec))))[-1 - elem]
    max_array = np.empty((1, max(len(str(round(min(y_vec)))), len(str(round(max(y_vec)))))), dtype="str")
    max_array[:] = ' '
    for elem in range(len(str(round(max(y_vec))))):
        max_array[:, elem] = list(str(round(max(y_vec))))[elem]
    array = np.vstack((max_array, middle_array, min_array))

    return array


def add_title(width_size: int, title: str) -> NDArray:
    """
    Generates a 2D array for the title of a plot or canvas
    :param width_size: Width of the plot or canvas
    :param title: Title of the plot or canvas
    :return: 2D array
    """
    title = title.upper()
    middle_array = np.empty((1, len(title)), dtype="str")
    for elem in range(len(title)):
        middle_array[:, elem] = list(title)[elem]
    if (width_size - len(title)) <= 0:
        array = middle_array
    else:
        lateral_array = np.empty((1, floor((width_size - len(title)) / 2)), dtype="str")
        lateral_array[:] = ' '
        if ((width_size - len(title)) / 2 % 2) == 0:
            array = np.hstack((lateral_array, middle_array, lateral_array))
        else:
            added_space = np.empty((1, 1), dtype="str")
            added_space[:] = ' '
            array = np.hstack((lateral_array, middle_array, lateral_array, added_space))

    return array


def norm_vec(vec: Union[List[Union[float]], NDArray]) -> List[Any]:
    """
    Normalizes an array using the min-max criteria
    :param vec: The vector to be normalized
    :return: Normalized array
    """
    normalized_vec = [(float(i) - min(vec)) / (max(vec) - min(vec)) for i in vec]

    return normalized_vec


def scale_vec(vec: Union[List[Union[float]], NDArray], scale_param: Union[int, float]) -> List[Union[Any]]:
    """
    Scales an array by multiplying all its elements by the same value
    :param vec: The vector to be scaled
    :param scale_param: The scaling parameter to use
    :return: Scaled array
    """
    scaled_vec = [floor(i * (scale_param - 1)) for i in vec]

    return scaled_vec


def norm_scale_2d_data(x_vec: Union[List[Union[float]], NDArray], y_vec: Union[List[Union[float]], NDArray],
                       width_size: int, height_size: int) -> Tuple[List[Union[float, Any]], List[Union[float, Any]]]:
    """
    Normalises and scales 2D data for e.g. a plot or canvas with x and y axes
    :param x_vec: The first array
    :param y_vec: The second array
    :param width_size: The maximum value we want to have in the first array
    :param height_size: The maximum value we want to have in the second array
    :return: Normalised and scaled 2D array
    """
    assert len(x_vec) == len(y_vec)

    vec_axis = [x_vec, y_vec]
    vec_size = [width_size, height_size]
    for dim in range(2):
        vec_axis[dim] = norm_vec(vec_axis[dim])
        vec_axis[dim] = scale_vec(vec_axis[dim], vec_size[dim])

    return vec_axis[0], vec_axis[1]


def print_data(x_vec: Union[List[Union[float]], NDArray], y_vec: Union[List[Union[float]], NDArray],
               display_grid: bool = False, width: int = 100, height: int = 15, add_xaxis_option: bool = True,
               add_yaxis_option: bool = True, legend: str = ' ', title: str = ' ') -> None:
    """
    Print 2D data
    :param x_vec: The data array to be displayed as the x axis
    :param y_vec: The data array to be displayed as the Y axis
    :param display_grid: if True, a grid is displayed
    :param width: Width of the plot or canvas
    :param height: Height of the plot or canvas
    :param add_xaxis_option: If True, adds the x axis extreme values
    :param add_yaxis_option: If True, adds the y axis extreme values
    :param legend: adds the legend of the plot or canvas within the x axis
    :param title: adds the title of the plot or canvas centered above it
    :return: Printed joint array displaying the 2D data
    """
    assert len(x_vec) == len(y_vec)

    empty_canvas = gen_canvas(height, width, display_grid)
    scaled_x, scaled_y = norm_scale_2d_data(x_vec, y_vec, width, height)

    pre_space = np.empty((1, len(str(height)) + 1), dtype="str")
    pre_space[:] = ' '
    print_chart(np.hstack((pre_space, add_title(width, title))))

    if add_yaxis_option:
        for point in range(len(scaled_x)):
            if display_grid:
                empty_canvas[scaled_y[point]][scaled_x[point]] = '*'
            else:
                empty_canvas[scaled_y[point]][scaled_x[point]] = '·'
        empty_canvas = np.flip(empty_canvas, axis=0)
        x_space = np.empty((height, 1), dtype="str")
        x_space[:] = ' '
        print_chart(np.hstack((add_yaxis_canvas(y_vec, height), x_space, empty_canvas)))

        if add_xaxis_option:
            y_space = np.empty((1, max(len(str(round(max(y_vec)))), len(str(round(min(y_vec))))) + 1), dtype="str")
            y_space[:] = ' '
            print_chart(np.hstack((y_space, add_xaxis_canvas(x_vec, width, legend))))

    elif not add_yaxis_option:
        for point in range(len(scaled_x)):
            if display_grid:
                empty_canvas[scaled_y[point]][scaled_x[point]] = '*'
            else:
                empty_canvas[scaled_y[point]][scaled_x[point]] = '·'
        empty_canvas = np.flip(empty_canvas, axis=0)
        print_chart(empty_canvas)

        if add_xaxis_option:
            print_chart(add_xaxis_canvas(x_vec, width, legend))


def plt_data(x_vec: List[Union[float]], y_vec: List[Union[float]]) -> None:
    """
    Additional function to check if the rest of the code works as expected which plots 2D data using matplotlib library
    :param x_vec: The data array to be displayed as the x axis
    :param y_vec: The data array to be displayed as the Y axis
    :return: Plot of the 2D data
    """

    plt.scatter(x_vec, y_vec)
    plt.show()


if __name__ == "__main__":

    # Example 1
    print("Example 1")
    scale = 0.1
    n = int(8 * math.pi / scale)
    x = [scale * i for i in range(n)]
    y = [math.sin(scale * i) for i in range(n)]
    print_data(x, y, display_grid=True, height=15, title="The sine function",
               legend="f(x) = sin(x), where 0 <= x <= 8π")
    plt_data(x, y)

    # Example 2
    print("Example 2")
    scale = 0.1
    n = int(2 * math.pi / scale)
    x = [scale * i for i in range(n)]
    y = [math.cos(scale * i) for i in range(n)]
    print_data(x, y, width=50, title="The cosine function", legend="f(x) = cos(x), where 0 <= x <= 2π")
    plt_data(x, y)

    # Example 3
    print("Example 3")
    y = [ord(c) for c in 'ASCII Plotter example']
    n = len(y)
    x = [i for i in range(n)]
    print_data(x, y, title="Plotting Random Data", legend="f(x) = random data")
    plt_data(x, y)


