from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from matplotlib import pyplot as plt


def within_polygon(m_point, m_polygon):
    point = Point(m_point[0], m_point[1])
    temp = []
    for i in m_polygon:
        temp.append((i[0], i[1]))
    polygon = Polygon(temp)
    # plt.plot(*polygon.exterior.xy)
    # plt.plot(point.x, point.y, 'ro')
    # plt.show()
    return polygon.contains(point)

# EXAMPLE
# P = [1, 1]
# S = [[0, 0], [20, 4], [0, 3]]
# print(within_polygon(P, S))
