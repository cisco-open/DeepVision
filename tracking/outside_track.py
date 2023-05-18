# Acknowledge

# Some codes are from (https://github.com/ifzhang/ByteTrack/blob/main/yolox/tracker/kalman_filter.py).
# Modified a little bit.

"""
This file is NOT complete.

I did not get a correct configuration to run the whole process, so some data structures are "guessed".

It that happens, I will write the guessed data structure, so any users can update them.
"""

import lap
import numpy as np
import scipy.linalg

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of freedom (contains values for N=1, ..., 9). 
Taken from MATLAB/Octave's chi2inv function and used as Mahalanobis gating threshold.
"""

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter:
    """
    A simple Kalman filter for tracking bboxes in images.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains:
    the bounding box center position (x, y),
    aspect ratio a,
    height h,
    and their respective velocities.

    Object motion follows a constant velocity model.
    """

    def __init__(self):
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current state estimate. These weights control
        # the amount of uncertainty in the model.

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """
        Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y), aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8 dimensional) of the new track.
            Unobserved velocities are initialized to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted state.
            Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # mean = np.dot(self._motion_mat, mean)
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """
        Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """
        Run Kalman filter prediction step (Vectorized version).

        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance metrics of the object states at the previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted state.
            Unobserved velocities are initialized to 0 mean.
        """

        # print("------------------------")
        # print("mean.shape: ", mean.shape)
        # print("cov.shape: ", covariance.shape)

        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """
        Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y) is the center position, a the aspect ratio,
            and h the height of the bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position = False, metric = "maha"):
        """
        Compute gating distance between state distribution and measurements. A suitable distance threshold can be
        obtained from "chi2inv95". If "only_position" is False, the chi-square distribution has 4 degrees of freedom,
        otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in format (x, y, a, h) where (x, y) is the bounding box
            center position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the squared Mahalanobis distance between
            (mean, covariance) and "measurements[i]".
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError("invalid distance metric")


class outside_tracker:
    """
    Each outside tracker is actually an "inside" tracker of each object, why it is called "outside" is because it is
    outside any existed tracker.

    Each outside tracker has a "unique object ID (UOI)", which is bound to the actual object in the world. Since we may
    have more than one existed tracker, and each of them may have their IDs. For example, in tracker 1, object 1 is
    assigned ID-0, but in tracker 2, the same object is assigned ID-2. This UOI is for solving this issue. It tries to
    make build a map between external and internal IDs.
    """

    shared_KF = KalmanFilter()

    def __init__(self, ID, label, xyah, score, life = 5):
        self.ID = ID
        self.label = label
        self.score = score

        self.KF = KalmanFilter()
        self.mean, self.cov = self.KF.initiate(xyah)

        # life is how many frames can a tracklet been assumed continuing without updates
        self.max_life = life
        self.life = life

    def predict(self):
        # single KF predicting
        self.retire()
        self.mean, self.cov = self.KF.predict(self.mean, self.cov)
        tlbr = outside_tracker_manager.xyah_to_tlbr(self.mean[:4])
        if tlbr[2] < 0 or tlbr[3] < 0:
            self.life = -1
        if tlbr[0] > 1280 or tlbr[1] > 720:
            self.life = -1

    def update(self, xyah, score):
        """
        "update" is for correcting the KF, say modifying its parameters.

        This will keep the mean and cov of the KF, and only the position of the bbox will be changed.
        """
        self.mean, self.cov = self.KF.update(self.mean, self.cov, xyah)
        self.score = score

    @property
    def xyah(self):
        return self.mean[: 4]

    def retire(self):
        self.life -= 1

    def activate(self, add_on = 5):
        self.life = min(self.max_life, self.life + add_on)

    def inactive(self):
        self.mean[7] = 0


class outside_tracker_manager:

    def __init__(self):
        self.internal_tracks = []
        self.outside_tracks_ref = dict()  # a reference for each outside_tracker, e.g., {id: idx of the outside_tracker}
        self.shared_KF = KalmanFilter()
        self.count = 0  # this is for assigning UOI

        # dictionary of external and internal IDs, {int: int}
        self.ID_mapping_1 = {}
        self.ID_mapping_2 = {}

        # an overall ID mapping, with external model ID mapped to the corresponding ID mapping, {model ID: ID mapping}
        self.ID_mapping = {}

        self.reasoner = None
        self.reID = None

    def reset(self):
        self.internal_tracks = []
        self.outside_tracks_ref = dict()
        self.shared_KF = KalmanFilter()

    def predict(self):
        for each in self.internal_tracks:
            if each is not None:
                each.predict()

    @staticmethod
    def bbox_ious(atlbrs, btlbrs):
        """
        Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2], say "tlbr".

        This code is from https://www.cnblogs.com/zhiyiYo/p/15586440.html.
        """
        A = atlbrs.shape[0]
        B = btlbrs.shape[0]

        xy_max = np.minimum(atlbrs[:, np.newaxis, 2:].repeat(B, axis=1),
                            np.broadcast_to(btlbrs[:, 2:], (A, B, 2)))
        xy_min = np.maximum(atlbrs[:, np.newaxis, :2].repeat(B, axis=1),
                            np.broadcast_to(btlbrs[:, :2], (A, B, 2)))

        inter = np.clip(xy_max - xy_min, a_min=0, a_max=np.inf)
        inter = inter[:, :, 0] * inter[:, :, 1]

        area_0 = ((atlbrs[:, 2] - atlbrs[:, 0]) * (
                atlbrs[:, 3] - atlbrs[:, 1]))[:, np.newaxis].repeat(B, axis=1)
        area_1 = ((btlbrs[:, 2] - btlbrs[:, 0]) * (
                btlbrs[:, 3] - btlbrs[:, 1]))[np.newaxis, :].repeat(A, axis=0)

        return inter / (area_0 + area_1 - inter)

    @staticmethod
    def iou_distance_util(atlbr, btlbr):
        ious = np.zeros((len(atlbr), len(btlbr)), dtype=np.float)  # iou similarity
        if ious.size == 0:
            return ious

        ious = outside_tracker_manager.bbox_ious(np.ascontiguousarray(atlbr, dtype=np.float),
                                                 np.ascontiguousarray(btlbr, dtype=np.float)
                                                 )

        cost_matrix = 1 - ious
        return cost_matrix

    @staticmethod
    def iou_distance(inactive_tracks, new_tracks):

        # TODO, this might be enhanced with the quality and label of each detection, when there are more labels

        tlbr_inactive_tracks = [outside_tracker_manager.xyah_to_tlbr(each.xyah) for each in inactive_tracks]
        tlbr_new_tracks = [outside_tracker_manager.xyah_to_tlbr(each[2]) for each in new_tracks]

        return outside_tracker_manager.iou_distance_util(tlbr_inactive_tracks, tlbr_new_tracks)

    @staticmethod
    def linear_assignment(cost_matrix, thresh = 0.8):
        # the return are just indices
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        matches, unmatched_a, unmatched_b = [], [], []
        cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
        matches = np.asarray(matches)
        return matches, unmatched_a, unmatched_b

    def create_new_track(self, new_track):  # for brand-new and for revision
        if new_track[0] in self.outside_tracks_ref:  # this object has been tracked before, but lost and deleted
            self.internal_tracks[self.outside_tracks_ref[new_track[0]]] = outside_tracker(new_track[0], new_track[1],
                                                                                          new_track[2], new_track[3])
        else:  # this id is brand-new
            self.internal_tracks.append(outside_tracker(new_track[0], new_track[1], new_track[2], new_track[3]))
            self.outside_tracks_ref.update({new_track[0]: len(self.internal_tracks) - 1})

    @staticmethod
    def tlbr_to_xyah(tlbr):
        x = tlbr[0]
        y = tlbr[1]
        h = tlbr[3] - y
        a = (tlbr[2] - x) / h
        return [x, y, a, h]

    @staticmethod
    def xyah_to_tlbr(xyah):
        tl_x = xyah[0]
        tl_y = xyah[1]
        br_x = tl_x + xyah[2] * xyah[3]
        br_y = tl_y + xyah[3]
        return [tl_x, tl_y, br_x, br_y]

    # def check_mappings(self, external_tracks, model_IDs):
    #     # all existed tracks, do predictions
    #     self.predict()
    #
    #     for i in range(len(external_tracks)):
    #         self.check_mapping(external_tracks[i], model_IDs[i])

    # def check_mapping(self, external_track, model_ID):
    #     """
    #     TODO, There might be some overlap between this function and the function self.step, but might be solved later.
    #     """
    #     tlbr_internal_rois = [self.xyah_to_tlbr(each.xyah) for each in self.internal_tracks]
    #     tlbr_external_rois = [each[:4] for each in external_track.get("bboxes", None)]
    #
    #     matches, unmatched_internal_rois, unmatched_external_rois = self.linear_assignment(
    #         self.iou_distance_util(tlbr_internal_rois, tlbr_external_rois))  # indices
    #
    #     # for matches
    #     for i_internal, i_external in matches:
    #
    #         internal_ID = self.internal_tracks[i_internal].ID
    #         external_ID = external_track.get("ids", None)[i_external]
    #
    #         if external_ID not in self.ID_mapping[model_ID]:  # first time of this external ID
    #             # mapped to an existed internal ID
    #             self.ID_mapping[model_ID].update({external_ID: internal_ID})
    #         else:  # this external ID is in the ID mapping
    #             # TODO, there should be some processing, but not here
    #             mapped_external_ID = self.ID_mapping[model_ID][external_ID]
    #             if mapped_external_ID != internal_ID:
    #                 # TODO, this is just the SIMPLEST method
    #                 # wrong match, just drop [but absolutely incorrect]
    #                 pass
    #             else:
    #                 # TODO, this is just the SIMPLEST method
    #                 # this looks good, nothing to do [but absolutely insufficient]
    #                 pass
    #
    #     # for unmatched_internal_rois
    #     # TODO, this is just the SIMPLEST method
    #     # no matches, say they just update themselves, which is done in self.predict()
    #
    #     # for unmatched_external_rois
    #     # TODO, this might be too much
    #     # we need to create EACH unmatched external track an internal track
    #     for i_external in unmatched_external_rois:
    #         # create new internal track
    #         ID = self.count
    #         label = external_track.get("labels", None)[i_external]
    #         tmp = external_track.get("bboxes", None)[i_external]
    #         tlbr = tmp[:4]
    #         score = tmp[-1]
    #         xyah = outside_tracker_manager.tlbr_to_xyah(tlbr)
    #         self.create_new_track([ID, label, xyah, score])
    #
    #         # create the mapping to this internal track
    #         external_ID = external_track.get("ids", None)[i_external]
    #         self.ID_mapping[model_ID].update({external_ID: ID})
    #
    #         # next internal ID
    #         self.count += 1

    # def ID_map(self, original_tracks, model_IDs):
    #     """
    #     To maintain the main structure, this function is to modify the "original_tracks".
    #
    #     It is to change the "ids" key in "original_tracks" with the inside ID. To make sure different external IDs are
    #     mapped to a unified internal ID.
    #     """
    #     self.check_mappings(original_tracks, model_IDs)
    #
    #     for i in range(len(original_tracks)):
    #         for j in range(len(original_tracks[i].get("ids", None))):
    #             original_tracks[i].get("ids", None)[j] = self.ID_mapping[model_ID][
    #                 original_tracks[i].get("ids", None)[j]]

    def step(self, outs_track):
        """
        outs_track: the detection + tracking results for each frame, generated by an existed tracker, processed by the
        function, results2outs. It is a dictionary with "ids", "labels" and "bboxes".

        Each is a np array, with index "i", the id will be outs_tracks.get("ids", None)[i]. I guess this "id" will be an
        integer.
        The label will be outs_tracks.get("labels", None)[i]. I guess this "label" will be a string.
        The bbox will be outs_tracks.get("bboxes", None)[i]. I guess this "bbox" will be a 4-tuple,
            e.g., (upper_left_x, upper_left_y, aspect ratio, height), say (x, y, a, h).
        And for the image processed, (0, 0) is the upper-left corner.
        """

        # for those tracks found by an existed tracker, if they are confident enough, I will take them, so use them to
        # update (correct) the KF in the corresponding outside_track.

        # the activeness is defined for each outside_track, that whether this track has been "updated" (corrected)
        # it it is "corrected", say modified by an existed tracker with a high confidence, it will be assumed settled.
        active_tracks = []
        inactive_tracks = []

        # except for inactive/active outside tracks, brand-new tracks might be introduced as well
        new_tracks = []

        # find active tracks, and they are done
        # TODO: though an existed tracker gives a tracking, that might also be incorrect.

        for i in range(len(outs_track.get("ids", None))):
            each_ID = outs_track.get("ids", None)[i]
            active_tracks.append(each_ID)
            tmp = outs_track.get("bboxes", None)[i]
            tlbr = tmp[:4]
            xyah = outside_tracker_manager.tlbr_to_xyah(tlbr)
            score = tmp[-1]

            if each_ID in self.outside_tracks_ref:  # a continued track, then update its KF
                each_track = self.internal_tracks[self.outside_tracks_ref[each_ID]]

                if each_track is None:  # if this is a continued lost track
                    self.create_new_track((
                        outs_track.get("ids", None)[i],
                        outs_track.get("labels", None)[i],
                        xyah,
                        score))
                else:
                    each_track.activate()
                    each_track.update(xyah, score)

            else:
                new_tracks.append((
                    outs_track.get("ids", None)[i],
                    outs_track.get("labels", None)[i],
                    xyah,
                    score))

        # find inactive tracks, "inactive" means currently the track is not updated (just for this frame)
        for each in self.internal_tracks:  # each is an outside_tracker
            if each is not None and each not in active_tracks:
                each.inactive()
                inactive_tracks.append(each)

        # all existed tracks, do predictions
        self.predict()

        # for those possibly new tracks, check whether they can be matched with some inactive tracks
        matches, unmatched_inactive_tracks, unmatched_new_tracks = self.linear_assignment(
            outside_tracker_manager.iou_distance(inactive_tracks, new_tracks))

        # process each match
        # matches: that means one "new track" is actually an "old inactive track", then the old one is continued
        # with the new one
        for i_inactive_tracks, i_new_tracks in matches:
            inactive_track = inactive_tracks[i_inactive_tracks]
            new_track = new_tracks[i_new_tracks]
            tmp = new_track[2]
            xyah = tmp[:4]
            score = new_track[3]

            inactive_track.update(xyah, score)  # update xyah and score
            inactive_track.activate()

        # process each unmatched_inactive_track
        # unmatched_inactive_tracks: previous tracks are not continued, they just "update themselves and life -1"
        # but this is already finished at the beginning

        # process each unmatched_new_track
        # unmatched_new_track: actual new_tracks
        for each in unmatched_new_tracks:  # this "each" is just an index
            self.create_new_track(new_tracks[each])

        # deleting these lost tracks
        for i in range(len(self.internal_tracks)):
            if self.internal_tracks[i] is not None and self.internal_tracks[i].life < 0:
                self.internal_tracks[i] = None

        return self.results2outs()

    def results2outs(self):
        ids = []
        labels = []
        bboxes = []

        for each in self.internal_tracks:
            if each is not None:
                ids.append(each.ID)
                labels.append(each.label)
                bboxes.append(outside_tracker_manager.xyah_to_tlbr(list(each.xyah)) + [each.score])

        outputs = {"ids": np.array(ids),
                   "labels": np.array(labels),
                   "bboxes": np.array(bboxes)}

        return outputs
