import numpy as np


def getData_EMG(user_id, session_nb, nb_repetition=10, transfer_learn=False,predict=False):
    # Parameters
    if transfer_learn:
        nb_gesture = 6
        nb_pts = 3000
    else:
        if predict:
            nb_gesture = 6
            nb_pts = 3000
        else:
            nb_gesture = 6
            nb_pts = 5000
    arm_used = "left"

    start_path = 'C:/Users/felix/OneDrive/Documents/phD/Conférences et démos/BioCAS/Real_time/user_000/session_' + session_nb + '/'  # ordi perso
    data_array = np.zeros((nb_gesture, nb_repetition, 64, nb_pts), dtype=np.int)
    for gest in range(nb_gesture):
        for rep in range(nb_repetition):
            path = start_path + user_id + "-" + session_nb + "-00" + str(gest) + "-00" + str(rep) + "-" + arm_used + ".csv"
            one_file = np.transpose(np.loadtxt(path, delimiter=','))
            data_array[gest, rep, :, :] = one_file[:, -nb_pts:]

    return data_array



#
# def getData_EMG(transfer_learn=False,predict=False):
#     # Parameters
#     user_id = "000"
#     session_nb = "004"
#     if transfer_learn:
#         nb_gesture = 6
#         nb_repetition = 3
#         nb_pts = 3000
#     else:
#         if predict:
#             nb_gesture = 6
#             nb_repetition = 1
#             nb_pts = 3000
#         else:
#             nb_gesture = 6
#             nb_repetition = 10
#             nb_pts = 5000
#     arm_used = "left"
#     #start_path = 'C:/Users/felix/OneDrive/Documents/SCHOOL/AUTOMNE2022/PAPER-TBioCAS/DATA/user_000/session_004/'  # ordi perso
#     start_path = 'C:/Users/felix/OneDrive/Documents/SCHOOL/HIVER2023/JIR2023/DATA/session_004/'  # ordi perso
#     # start_path = 'C:/Users/fecha24/OneDrive/Documents/SCHOOL/ETE2022/Projet/Dataset/user_001/session_000/'  # ordi UL
#     data_array = np.zeros((nb_gesture, nb_repetition, 64, nb_pts), dtype=np.int)
#     for gest in range(nb_gesture):
#         for rep in range(nb_repetition):
#             path = start_path + user_id + "-" + session_nb + "-00" + str(gest) + "-00" + str(rep) + "-" + arm_used + ".csv"
#             one_file = np.transpose(np.loadtxt(path, delimiter=','))
#             data_array[gest, rep, :, :] = one_file[:, -nb_pts:]
#
#     return data_array
