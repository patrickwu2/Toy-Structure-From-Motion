import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


class TwoViewRecon:
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.matcher = cv2.BFMatcher()

    def run(self, img1, img2, K):
        self.K = K
        # feature matching
        pts1, pts2 = self.feature_matching(img1, img2)
        # pose estimation
        R1, t1 = np.eye(3, 3), np.zeros((3, 1))
        R2, t2 = self.get_pose(pts1, pts2)
        # triangulation
        pcl = self.triangulation(pts1, pts2, R1, R2, t1, t2)
        return pcl
        vispcl(pcl)

    def feature_matching(self, img1, img2):
        # SIFT feature extraction
        feat1 = self.get_key_points(img1)
        feat2 = self.get_key_points(img2)
        # Matching Algorithm
        pts1, pts2 = self.match(feat1, feat2)
        return pts1, pts2

    def get_pose(self, pts1, pts2):
        # compute Fundamental Matrix, Essential Matrix
        F, mask = self.compute_F_matrix(pts1, pts2)
        E = self.compute_E_matrix(F, self.K)
        _, R, t, _ = cv2.recoverPose(E, pts1[mask], pts2[mask], self.K)
        return R, t

    def get_key_points(self, img):
        return self.sift.detectAndCompute(img, None)

    def match(self, feat1, feat2):
        kp1, des1 = feat1
        kp2, des2 = feat2
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        pts1 = []
        pts2 = []
        for m, n in matches:
            # if m.distance < 0. * n.distance:
            if True:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
        return np.array(pts1), np.array(pts2)

    def compute_F_matrix(self, pts1, pts2):
        return cv2.findFundamentalMat(
                    pts1, pts2, cv2.FM_RANSAC,
                    1.0, 0.999
                )

    def compute_E_matrix(self, F, K):
        return K.T.dot(F.dot(K))

    def triangulation(self, pts1, pts2, R1, R2, t1, t2):
        img1ptsHom = cv2.convertPointsToHomogeneous(pts1)[:, 0, :]
        img2ptsHom = cv2.convertPointsToHomogeneous(pts2)[:, 0, :]

        img1ptsNorm = (np.linalg.inv(self.K).dot(img1ptsHom.T)).T
        img2ptsNorm = (np.linalg.inv(self.K).dot(img2ptsHom.T)).T

        img1ptsNorm = cv2.convertPointsFromHomogeneous(img1ptsNorm)[:, 0, :]
        img2ptsNorm = cv2.convertPointsFromHomogeneous(img2ptsNorm)[:, 0, :]

        pts4d = cv2.triangulatePoints(
                        np.hstack((R1, t1)), np.hstack((R2, t2)),
                        img1ptsNorm.T, img2ptsNorm.T
                    )
        pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:, 0, :]
        return pts3d


def vispcl(pts3d):
    # print(pts3d[:30])
    # exit()
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter3D(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], s=2)

    ax.set_xlim(left=-20, right=20)
    ax.set_ylim(bottom=-20, top=20)
    ax.set_zlim(bottom=-20, top=20)
    plt.show()


def read_img(fname):
    img = cv2.imread(fname)
    return img


def read_mat(fname):
    return np.loadtxt(fname)


if __name__ == "__main__":
    cls = TwoViewRecon()
    fname1 = "fountain-P11/images/0000.jpg"
    fname2 = "fountain-P11/images/0001.jpg"
    Kfname = "fountain-P11/images/K.txt"
    img1 = read_img(fname1)
    img2 = read_img(fname2)
    K = read_mat(Kfname)
    cls.run(img1, img2, K)
