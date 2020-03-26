import numpy as np
import cv2

class KF():

    def __init__(self, A = None, H = None, P = None, Q = None, R = None, dt =
                 None, x0 = None):
        self.A = A
        self.B = 0 ####Suponiendo que no hay 0
        self.H = H
        self.P = P
        self.Q = Q
        self.R = R
        self.dt = dt
        self.x = x0
        self.n = A.shape[1]
#        self.W = 0

    def predecir(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, 0)
#        print("Pred: ", self.x)
#        input("Press enter")
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
#        print("Cov: ", self.P)
#        input("Press enter")
        return self.x
    def actualizar(self, Y):
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(np.dot(np.dot(self.H, self.P), self.H.T) + self.R))
#        print("KG: ", K)
#        input("Press enter")
        self.x = self.x + np.dot(K, (Y - np.dot(self.H, self.x)))
#        print("XK: ", self.x)
#        input("Press enter")
        self.P = np.dot((np.eye(self.n) - (np.dot(K, self.H))), self.P)
#        print("New cov: ", self.P)
#        input("Press enter")
        return self.x


def main():

    dt = 1

    #Matriz de estados A

    A = np.array([[1, 0, 0, 0, dt, 0, 0, 0],
                 [0, 1, 0, 0, 0, dt, 0, 0],
                 [0, 0, 1, 0, 0, 0, dt, 0],
                 [0, 0, 0, 1, 0, 0, 0, dt],
                 [0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1]])

    #Matriz de medidas H
    H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0]])

    #Deviation Matrix
    a = np.true_divide((A - np.dot(np.ones((8, 8)), A)), (A.shape[1]))
    P = np.dot(a.T, a)
#    P = np.diag(100*np.ones(8))
#    P = np.diag(np.diag(P))
#    C = np.eye(H.shape[1])

    R = np.array([[0.65, 0, 0, 0],
                  [0, 0.65, 0, 0],
                  [0, 0, 0.65, 0],
                  [0, 0, 0, 0.65]])

    Q = np.array([[0.25, 0, 0, 0, 0.5, 0, 0, 0],
                  [0, 0.25, 0, 0, 0, 0.5, 0, 0],
                  [0, 0, 0.25, 0, 0, 0, 0.5, 0],
                  [0, 0, 0, 0.25, 0, 0, 0, 0.5],
                  [0.5, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0.5, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0.5, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0.5, 0, 0, 0, 1]])
    Q = Q*0.01
    Y = np.zeros((4, 1))
    x0 = np.array(np.zeros((8, 1)))
    x0[:, 0] = (0, 40, 100, 70, 1, 1, 1, 1)

    cap = cv2.VideoCapture("test_salida.avi")
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    filename = '/home/camilo685/Desktop/salidakalman.mp4'
    Output = cv2.VideoWriter(filename, codec, 30, (frame_width,frame_height))
    fgbg = cv2.createBackgroundSubtractorKNN()

    obj1 = KF(A, H, P, Q, R, dt, x0)

    XPre = np.array(np.zeros((8, 1)))
    XKal = np.array(np.zeros((8, 1)))

    while(1):
        ret, frame = cap.read()
        if ret:
            fgmask = fgbg.apply(frame)
            contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < 5000 or cv2.contourArea(c) > 200000:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
#                print("x: ", x, "y: ", y, "w: ", w, "h: ", h)
#                input("Press enter")
                Y[:, 0] = (x, y, w, h)
                XPre = obj1.predecir()
                XKal = obj1.actualizar(Y)
                (xk, yk, wk, hk, v1, v2, v3, v4) = XKal
                (xp, yp, wp, hp, v1, v2, v3, v4) = XPre

				#Deteccion, verde
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #salida kalman, azul
                cv2.rectangle(frame, (xk, yk), (xk + wk, yk + hk), (255, 0, 0), 2)
#                cv2.rectangle(frame, (xp, yp), (xp + wp, yp + hp), (0, 0, 255), 2)
                Output.write(frame)
                cv2.imshow("frame", frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ ==  "__main__":
	main()
