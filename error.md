1. PnP :이 문제는 3D 물체와 이미지 투영 사이의 알고 있는 3D에서 2D 로의 대응점으로부터 캘리브레이션 된 카메라의 위치와 방향을 추정하는 것이다
-> [참고논문](https://www.koreascience.or.kr/article/JAKO201911338887336.pdf)
2. Perspective-n-Point 문제가 무엇인가요?
답:  
Perspective-n-Points 문제는 2D-3D correspondence가 존재할 때, 카메라에서부터 world space에 대한 변환관계를 알아내려고 하는 문제이다. 간단하게 이야기해서, perspective (물체를 바라보고있을 때), n-point (n개의 2D-3D point correspondence가 있다면)로 생각할 수 있다.  
보통 2D 정보는 image feature detection에서 나오고, 3D 정보는 3D reconstruction 단계에서 저장된 point cloud이다. 2D image feature의 경우 feature descriptor를 가지고 있고, 3D point cloud도 각각의 포인트마다 descriptor를 저장하고 있기 때문에 매칭이 가능하다. 물론 다른 방법으로도 매칭할 수 있다.  
PnP 문제를 풀기위해 필요한 최소 correspondence 수는 3개이다. 이를 사용하는 알고리즘이 P3P인데, 종종 방향에 대해 ambiguity가 생기기 때문에 하나를 더 추가해서 사용하기도 한다. 이 외로 EPnP 또는 UPnP와 같은 방법들도 있다.  
위에서 방금 소개했던 방식들은 closed-form으로 계산을 하기 때문에, 2D 정보나 3D 정보에 노이즈가 껴있는 경우 정확한 값이 나오지 않거나, 완전히 다른 값이 나타날 수 있다. 이 때문에 정확한 값을 얻기 위해서는 더 많은 correspondence들을 추출해낸 후 RANSAC을 돌리는 방법이 많이 사용된다. 또 다른 방법으로는 optimization을 사용해서 2D정보와 3D 정보를 변형하는 방법도 있다   .
OpenCV에서 사용하고 있는 PnP 솔버 함수에는 *solvePnP()*와 *solvePnPRANSAC()*이 있다. 후자의 경우가 minimal set 방식을 이용해서 푸는 방식이다. 전자의 경우에는 RANSAC을 사용하지 않고, 문제를 non-linear optimization 방식으로 바꿔버린다. 그렇기 때문에, 얻어진 3D pose로 reprojection을 했을 때 정확하게 작동하지 않을 수 있고, 또 outlier correspondence가 있을 때 위험할 수도 있다. 
