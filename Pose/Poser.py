import cv2
import numpy as np
from Pose.PoseModel import ModelData

KERNEL = (3, 3)
SIGMA_X = 0
SIGMA_Y = 0
THRESHOLD = 0.1
N_INTERP_SAMPLES = 10
CONFIDENCE_THRESHOLD = 0.7


class PoseCalculator:
    def __init__(self, model_info: ModelData, model_output: np.ndarray, image: np.ndarray):
        self.model_info = model_info
        self.model_output = model_output
        self.image = image
        self.detected_key_points = list()
        self.key_point_list = np.zeros((0, 3))

    def get_plottable_key_points(self) -> np.ndarray:
        dkp, kl = self.get_detected_key_points()
        print(f"KL = {kl}")
        valid, invalid = self.get_valid_pairs()
        return self.get_personwise_key_points(valid, invalid)

    def get_detected_key_points(self) -> tuple:
        keypoint_id = 0

        for part in range(self.model_info.num_of_kp):
            prob_map = self.model_output[0, part, :, :]
            prob_map = cv2.resize(prob_map, (self.image.shape[1], self.image.shape[0]))
            keypoints = self.get_key_points(prob_map)
            # print("Keypoints - {} : {}".format(self.model_info.key_points[part], keypoints))

            keypoints_with_id = []
            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                self.key_point_list = np.vstack([self.key_point_list, keypoints[i]])
                keypoint_id += 1

            self.detected_key_points.append(keypoints_with_id)

        return self.detected_key_points.copy(), self.key_point_list.copy()

    def get_key_points(self, prob_map: np.ndarray) -> list:
        map_smooth = cv2.GaussianBlur(prob_map, KERNEL, SIGMA_X, 0)
        map_mask = np.uint8(map_smooth > THRESHOLD)
        key_points = []
        contours, _ = cv2.findContours(map_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            blob_mask = np.zeros(map_mask.shape)
            blob_mask = cv2.fillConvexPoly(blob_mask, cnt, 1)
            masked_prob_map = map_smooth * blob_mask
            _, max_val, _, max_loc = cv2.minMaxLoc(masked_prob_map)
            key_points.append(max_loc + (prob_map[max_loc[1], max_loc[0]],))

        return key_points

    def get_valid_pairs(self) -> tuple:
        valid_pairs = []
        invalid_pairs = []

        for k in range(len(self.model_info.map_index)):
            # A->B constitute a limb
            paf_a = self.model_output[0, self.model_info.map_index[k][0], :, :]
            paf_b = self.model_output[0, self.model_info.map_index[k][1], :, :]
            paf_a = cv2.resize(paf_a, (self.image.shape[1], self.image.shape[0]))
            paf_b = cv2.resize(paf_b, (self.image.shape[1], self.image.shape[0]))

            # Find the keypoints for the first and second limb
            candA = self.detected_key_points[self.model_info.pose_pairs[k][0]]
            candB = self.detected_key_points[self.model_info.pose_pairs[k][1]]
            na = len(candA)
            nb = len(candB)

            # If keypoints for the joint-pair is detected
            # check every joint in candA with every joint in candB
            # Calculate the distance vector between the two joints
            # Find the PAF values at a set of interpolated points between the joints
            # Use the above formula to compute a score to mark the connection valid

            if na != 0 and nb != 0:
                valid_pair = np.zeros((0, 3))
                for i in range(na):
                    max_j = -1
                    max_score = -1
                    found = 0
                    for j in range(nb):
                        # Find d_ij
                        d_ij = np.subtract(candB[j][:2], candA[i][:2])
                        norm = np.linalg.norm(d_ij)
                        if norm:
                            d_ij = d_ij / norm
                        else:
                            continue
                        # Find p(u)
                        interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=N_INTERP_SAMPLES),
                                                np.linspace(candA[i][1], candB[j][1], num=N_INTERP_SAMPLES)))
                        # Find L(p(u))
                        paf_interp = []
                        for k in range(len(interp_coord)):
                            paf_interp.append([paf_a[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                               paf_b[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))]])
                        # Find E
                        paf_scores = np.dot(paf_interp, d_ij)
                        avg_paf_score = sum(paf_scores) / len(paf_scores)

                        # Check if the connection is valid
                        # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                        if (len(np.where(paf_scores > THRESHOLD)[0]) / N_INTERP_SAMPLES) > CONFIDENCE_THRESHOLD:
                            if avg_paf_score > max_score:
                                max_j = j
                                max_score = avg_paf_score
                                found = 1
                    # Append the connection to the list
                    if found:
                        valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], max_score]], axis=0)

                # Append the detected connections to the global list
                valid_pairs.append(valid_pair)
            else:  # If no keypoints are detected
                invalid_pairs.append(k)
                valid_pairs.append([])
        return valid_pairs, invalid_pairs

    def get_personwise_key_points(self, valid_pairs, invalid_pairs) -> np.ndarray:
        # the last number in each row is the overall score
        print(valid_pairs)
        personwiseKeypoints = -1 * np.ones((0, 19))

        for k in range(len(self.model_info.map_index)):
            if k not in invalid_pairs:
                partAs = valid_pairs[k][:,0]
                partBs = valid_pairs[k][:,1]
                indexA, indexB = np.array(self.model_info.pose_pairs[k])

                for i in range(len(valid_pairs[k])):
                    found = 0
                    person_idx = -1
                    for j in range(len(personwiseKeypoints)):
                        print(f"p: {personwiseKeypoints[j][indexA]} == {partAs[i]}")
                        if personwiseKeypoints[j][indexA] == partAs[i]:
                            person_idx = j
                            found = 1
                            break

                    if found:
                        personwiseKeypoints[person_idx][indexB] = partBs[i]
                        personwiseKeypoints[person_idx][-1] += (
                                self.key_point_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2])

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(19)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        # add the keypoint_scores for the two keypoints and the paf_score
                        row[-1] = sum(self.key_point_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                        personwiseKeypoints = np.vstack([personwiseKeypoints, row])
        return personwiseKeypoints
