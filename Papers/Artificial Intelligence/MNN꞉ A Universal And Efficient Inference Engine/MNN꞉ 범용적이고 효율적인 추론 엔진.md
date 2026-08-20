# MNN: 범용적이고 효율적인 추론 엔진

Bin Zou<sup>1</sup>, Chengfei Lv<sup>1</sup>, Huan Wang<sup>2</sup>, Lichuan Wang<sup>1</sup>, Tianhang Yu<sup>1</sup>, Xiaotang Jiang<sup>1</sup>, Yafeng Yang<sup>1</sup>, Yiliu Chen<sup>1</sup>, Yu Cai<sup>1</sup>, Zhihua Wu<sup>1</sup>, Ziqi Wu<sup>1</sup>, Zongyang Cui<sup>1</sup>

<sup>1</sup> Alibaba Group, 중국 항저우.<br/>
<sup>2</sup> 미국 보스턴 Notheastern University 전기·컴퓨터공학과. 교신 저자: Ziqi Wu <mingyi.wzq@alibaba-inc.com>.

2020년 미국 텍사스주 오스틴에서 열린 제3회 MLSys Conference 논문집. 저작권 © 2020 저자.

## 초록

모바일 장치에 딥러닝 모델을 배포하는 일은 최근 점점 더 많은 관심을 받고 있다. 그러나 장치용 고효율 추론 엔진을 설계하려면 모델 호환성, 장치 다양성, 자원 제약이라는 큰 과제를 해결해야 한다. 이러한 과제에 대응하기 위해 우리는 모바일 애플리케이션에 특화된 범용적이고 효율적인 추론 엔진인 Mobile Neural Network (MNN)를 제안한다. 본 논문에서 MNN의 기여는 다음과 같다. (1) 런타임 최적화를 수행하는 사전 추론(pre-inference)이라는 메커니즘을 제시한다. (2) 최적의 계산 성능을 달성하도록 연산자 커널을 철저히 최적화한다. (3) 하이브리드 스케줄링을 가능하게 하면서 엔진을 경량으로 유지하는 백엔드 추상화 모듈을 도입한다. 광범위한 벤치마크 실험은 MNN이 널리 사용되는 다른 경량 딥러닝 프레임워크보다 우수한 성능을 보인다는 사실을 입증한다. MNN은 다음 주소에서 공개되어 있다: https://github.com/alibaba/MNN.

## 1. 서론

딥러닝은 컴퓨터 비전, 사용자 의도 인식(Guo et al., 2019), 자율주행(LeCun et al., 2015)을 비롯한 다양한 인공지능 작업에서 사실상의 표준 방법이 되었다. 이제 엣지 장치(예: 스마트폰, IoT 장치, 웨어러블 장치)가 널리 보급되면서 엣지 장치, 특히 모바일 장치에서의 딥러닝이 점점 더 주목받고 있다(Shi et al., 2016). 모바일에서 딥러닝을 실행하면 낮은 지연 시간, 개인정보 보호, 개인화 서비스 등 여러 이점이 있다. 온디바이스 딥러닝 기술을 충분히 활용하기 위해 모바일 장치에 특화된 추론 엔진이 개발되었고 모바일 애플리케이션에서 폭넓게 사용되고 있다. 그 예로 TF-Lite (Google, 2017a)(Google, 2017a), NCNN (Tencent, 2017), CoreML (Apple, 2017) 등이 있다.

모바일 추론 엔진이 직면한 주요 과제는 모델 호환성, 장치 다양성, 자원 제약이라는 세 측면으로 분류할 수 있다.

**(1) 모델 호환성.** 모바일 장치에 배포되는 대부분의 딥러닝 모델은 TensorFlow (Abadi et al., 2016), PyTorch (Paszke et al., 2017), Caffe (Jia et al., 2014), CNTK (Yu et al., 2014), MXNet (Chen et al., 2015)과 같이 잘 알려진 딥러닝 프레임워크에서 학습된다. 따라서 서로 다른 형식과 서로 다른 연산자를 지원하는 모델 호환성은 추론 엔진의 기본 요건이다. 더 중요하게는 앞으로 등장할 새로운 연산자를 지원할 수 있도록 엔진이 적절한 확장성도 제공해야 한다.

**(2) 장치 다양성.** 거의 모든 유명 모바일 애플리케이션은 단일 코어 CPU를 탑재한 저사양 장비부터 Apple Neural Engine(ANE) 같은 보조 프로세서를 갖춘 고사양 장비까지 매우 다양한 장치에서 폭넓게 사용된다. 여러 장치에서 높은 성능을 달성하려면 모바일 추론 엔진은 하드웨어 아키텍처뿐 아니라 ARM Mali GPU나 Qualcomm Adreno GPU 같은 장치 공급업체의 특성까지 고려해야 한다. 또한 서로 다른 운영체제(Android/iOS/임베디드 OS)와 Android GPU를 위한 서로 다른 솔루션 표준(OpenCL (Khronos, 2009)/OpenGL (Khronos, 1992)/Vulkan (Khronos, 2015)) 등 소프트웨어 다양성 문제도 잘 처리할 수 있어야 한다.

**(3) 자원 제약.** 하드웨어가 빠르게 발전하고 있음에도 모바일 장치의 메모리와 계산 능력은 여전히 제한적이며, 데스크톱 및 서버에 비해 몇 자릿수나 낮다.

위 과제를 종합하면 좋은 모바일 추론 엔진은 다음 두 속성을 갖추어야 한다. (1) 모델 호환성과 장치 다양성을 모두 해결하는 *범용성(universality)*, (2) 장치에서 모델을 높은 성능으로 추론하면서 메모리와 에너지 소비를 가능한 한 적게 사용하는 *효율성(efficiency)*이다.

이러한 속성을 충족하기 위해 우리는 *Mobile Neural Network (MNN)*라는 새로운 모바일 추론 엔진을 소개한다. 본 연구의 기여를 요약하면 다음과 같다.

* 온라인 비용 평가와 최적 방식 선택을 통해 런타임 최적화를 수행하는 사전 추론 메커니즘을 제시한다.

* 개선된 알고리즘과 데이터 레이아웃을 활용해 널리 사용되는 일부 연산의 성능을 더욱 높이는 심층적인 커널 최적화를 제공한다.

* 하이브리드 스케줄링을 지원하면서 엔진 자체를 가능한 한 경량으로 유지하는 백엔드 추상화 모듈을 제안한다. 애플리케이션에 MNN을 통합했을 때 바이너리 크기는 400∼600KB만 증가한다.

MNN은 이미 많은 모바일 애플리케이션에서 폭넓게 채택되었다는 점에 주목할 필요가 있다. 아울러 우리는 커뮤니티를 풍성하게 하고 더 많은 사람이 개선에 참여할 수 있도록 전체 프로젝트를 오픈 소스로 공개한다.

![](./Figure%2001.png)<br/>
**그림 1.** 제안하는 Mobile Neural Network의 개요.

## 2. 관련 연구

최근 온디바이스 딥러닝에 대한 수요가 증가하면서 특히 여러 주요 기업을 중심으로 모바일 추론 솔루션이 큰 관심을 받고 있다.

CoreML은 머신러닝 모델을 iOS 소프트웨어 애플리케이션에 통합하기 위한 Apple의 프레임워크로, CPU, GPU, ANE와 같은 여러 하드웨어를 활용할 수 있다. Android 스마트폰에 대해서도 Google은 온디바이스 추론을 위한 자체 솔루션인 ML Kit과 Neural Networks API(NNAPI)를 제공한다(Google, 2016). 그러나 이러한 솔루션의 가장 큰 단점은 범용성이 제한된다는 점이다. 예를 들어 CoreML은 iOS 11 이상을 요구하고 NNAPI는 Android 8.1 이상을 요구하므로, 기존의 많은 휴대전화와 임베디드 장치가 지원 대상에서 제외된다.

2017년 말 Google은 효율적인 모바일 딥러닝 프레임워크인 TensorFlow Lite(TF-Lite)를 공개했다(Google, 2017a). TF-Lite는 휴대전화와 임베디드 장치처럼 계산 능력이 낮은 장치에 맞게 최적화되어 있다. 거의 같은 시기에 Facebook은 개발자가 모바일 장치에 딥러닝 모델을 배포할 수 있도록 Caffe2를 공개했다(Paszke et al., 2017). 두 프레임워크 모두 다양한 장치를 지원하며, 이미 이를 기반으로 많은 애플리케이션이 개발되었다. 그러나 TF-Lite나 Caffe2를 사용한 온디바이스 추론은 때때로 경량화라는 목표와 충돌할 수 있다. 예를 들어 TF-Lite는 부동소수점 연산을 가속하기 위해 Accelate, Eigen, OpenBLAS 라이브러리를 사용하고, Caffe2는 행렬 연산을 가속하기 위해 Eigen에 의존한다. 이러한 의존성과 함께 모바일 딥러닝 프레임워크를 통합하면 모바일 애플리케이션의 바이너리 크기가 커지고 불필요한 오버헤드가 발생한다.

위 문제를 해결하려는 많은 노력이 이루어졌다. NCNN (Tencent, 2017), MACE (Xiaomi, 2018),
Anakin (Baidu, 2018)이 대표적이다. 이들은 우리가 *수동 탐색(manual search)* 또는 *비자동 탐색(non-automated search)*이라고 부르는 패러다임을 따른다. 이 패러다임에서는 외부 라이브러리에 의존하지 않고 세심하게 설계된 어셈블리 명령어로 연산자를 사례별로 최적화한다. 예를 들어 스트라이드가 1인 3 × 3 합성곱을 위해 하나의 전용 프로그램 함수를 구현하고, 스트라이드가 2인 3 × 3 합성곱을 위해서는 또 다른 함수를 별도로 구현해야 한다. 이러한 구현 방식은 모바일 추론 엔진을 가볍고 효율적으로 만들 수 있다. 그러나 사례별 최적화는 시간이 많이 들며 새롭게 등장하는 모든 연산자를 포괄하기 어렵다.

수동 탐색과 뚜렷이 대조되는 반대편에는 TVM (DMLC, 2016)이 선도한 *자동 탐색(automated search)*이라는 또 다른 철학이 있다. TVM은 다양한 딥러닝 모델을 엔드투엔드 방식으로 라이브러리로 컴파일할 수 있는 개방형 딥러닝 컴파일러 스택이다. TVM은 중복 의존성 문제를 해결할 뿐 아니라 모델과 백엔드에 맞춤화된 그래프 수준 및 연산자 수준 최적화도 제공한다. 그 결과 TVM은 매우 고무적인 성능을 보이며 모델과 장치 다양성 양쪽 측면에서 확장성이 뛰어나다. 그러나 이러한 장점에는 대가가 따른다. TVM이 생성하는 런타임 라이브러리는 *모델별(model-specific)*이다. 즉, 모델을 업데이트하려면—많은 AI 기반 소프트웨어 애플리케이션에서 매우 흔하고 빈번한 일이다—코드를 다시 생성하고 새 버전을 배포해야 한다. 이 메커니즘은 비용 부담을 일으키며 모바일 애플리케이션에서는 때때로 실용적이지 않다. 본 논문에서는 모바일 배포에서 더 높은 범용성과 더 나은 성능을 제공하는 반자동 탐색 아키텍처를 개발하고자 한다.

그 밖에도 온디바이스 딥러닝과 관련된 연구가 있다. 예를 들어 계산 그래프 DSL(Domain-Specific Language)(Abadi et al., 2016; Bastien et al., 2012)을 이용한 그래프 수준 최적화, 연산자 융합 및 치환(Google, 2017b; Wei et al., 2017) 등이 있다. 이러한 연구는 본 논문의 기여와 서로 독립적이며 MNN은 그중 일부를 참고한다.

## 3. Mobile Neural Network (MNN)

### 3.1. MNN 개요

MNN의 기본 워크플로는 오프라인 변환과 온디바이스 추론이라는 두 부분으로 구성된다. 그림 2는 MNN의 전체 아키텍처를 요약하며, 이 절에서는 각 구성 요소를 살펴보면서 MNN이 워크플로를 어떻게 최적화하는지 간략히 설명한다.

![](./Figure%2002.png)<br/>
**그림 2.** 제안하는 모바일 추론 엔진 Mobile Neural Network (MNN)의 세부 아키텍처.

오프라인 변환 단계에서는 먼저 변환기가 서로 다른 딥러닝 프레임워크의 모델을 입력으로 받아 MNN의 자체 모델 형식(`.mnn`)으로 변환한다. 이와 동시에 연산자 융합(Ashari et al., 2015), 치환, 모델 양자화(Rastegari et al., 2016) 같은 기본적인 그래프 최적화를 수행한다.

온디바이스 추론에는 사전 추론, 연산자 수준 최적화, 백엔드 추상화라는 세 가지 모듈이 관여한다. 각 연산자에 대해 사전 추론 모듈은 입력 크기와 커널 형태 같은 정보에 코어 수와 하드웨어 가용성 같은 백엔드 속성을 결합하는 *비용 평가 메커니즘*을 제공한다. 이 메커니즘은 후보 방식 풀에서 최적의 계산 방식을 동적으로 결정한다. 이어서 연산자 수준 최적화 모듈은 고급 알고리즘과 SIMD(Single Instruction Multiple Data), 파이프라이닝 같은 기법을 함께 사용하여 성능을 더욱 높인다.

또한 MNN은 여러 하드웨어 아키텍처를 백엔드로 지원한다. 모든 하드웨어 사양에 들어맞는 단일 표준은 없으므로 MNN은 OpenCL, OpenGL, Vulkan, Metal 같은 서로 다른 소프트웨어 솔루션을 지원한다. 모든 백엔드는 독립적인 구성 요소로 구현되며, 백엔드 추상화 모듈이 통일된 인터페이스 집합을 제공하여 이기종 백엔드의 메모리 관리 같은 내부 세부 사항을 숨긴다.

제안한 아키텍처를 통해 온디바이스 추론에서 높은 성능을 달성할 뿐 아니라 TPU, FPGA 등 새롭게 등장하는 더 많은 백엔드로 MNN을 쉽게 확장할 수 있다. 이 절의 나머지 부분에서는 MNN 아키텍처를 더 자세히 설명한다.

### 3.2. 사전 추론

사전 추론은 제안하는 반자동 탐색 아키텍처의 핵심 부분이다. 많은 딥러닝 애플리케이션에서 입력 크기가 일반적으로 고정되어 있거나 목표 크기로 전처리할 수 있다는 공통적인 특성을 활용한다. 이를 바탕으로 실제 추론을 수행하기 *전에* 메모리 사용량과 계산 비용을 결정할 수 있다. 따라서 메모리 사전 할당과 재사용 같은 최적화 기법을 적용하여 성능을 더욱 향상할 수 있다. 사전 추론은 계산 방식 선택과 준비-실행 분리라는 두 부분으로 나눌 수 있다.

**계산 방식 선택.** 알고리즘 구현과 백엔드 특성을 모두 고려하여 후보 방식 풀에서 최적 방식을 선택하는 *비용 평가 메커니즘*을 제안한다.

$$
\begin{array}{cr}
C_{\mathrm{total}} = C_{\mathrm{algorithm}} + C_{\mathrm{backend}},
&
\qquad \text{(1)}
\end{array}
$$

여기서 $C$는 비용을 나타낸다.

(1) 합성곱 방식 풀을 예로 들면, 일반적으로 슬라이딩 윈도와 Winograd(Lavin & Gray, 2016)라는 두 가지 빠른 구현 알고리즘 가운데 하나를 선택할 수 있다. 기본 아이디어는 서로 다른 합성곱 설정에 따라 계산 비용을 최소화하는 알고리즘을 동적으로 선택하는 것이다. 따라서 $C_{\mathrm{algorithm}}$ 항을 최소화하는 최적 계산 방식은 다음과 같이 결정할 수 있다.

1. 커널 크기가 $k = 1$이면 단순한 행렬 곱셈이므로 Strassen 알고리즘(Strassen, 1969)을 적용한다.

2. 커널 크기가 $k > 1$이면 Winograd를 사용하여 합성곱을 행렬 곱셈으로 변환한다. Winograd 합성곱에는 출력 타일 크기 $n$에 따른 여러 하위 선택지가 있으며, 그 산술 비용은 다음과 같이 나타낼 수 있다. 이 비용을 바탕으로 비용 $C$를 최소화하는 최적 출력 타일 크기 $\hat{n}$을 선택한다.

$$
\begin{array}{cr}
\begin{aligned}
C(n) &= 2i_c(n+k-1)^3 \\
&\quad + i_c o_c(n+k-1)^2 \\
&\quad + n(n+k-1)(2n+k-1), \\
\Rightarrow\ \hat{n} &= \underset{n}{\arg\min}\, C(n).
\end{aligned}
&
\qquad \text{(2)}
\end{array}
$$

3. 이어서 $\hat{n}$이 1이면 슬라이딩 윈도를 선택하고, 그렇지 않으면 Winograd 합성곱을 사용한다.

합성곱을 위한 이 비용 평가 메커니즘은 다음과 같이 요약할 수 있다.

$$
\begin{array}{cr}
\mathrm{Scheme} =
\begin{cases}
\text{sliding window}, & \text{if } k > 1 \text{ and } \hat{n} = 1, \\
F(\hat{n} \times \hat{n}, k \times k), & \text{if } k > 1 \text{ and } \hat{n} > 1.
\end{cases}
&
\qquad \text{(3)}
\end{array}
$$

(2) 다음 문제는 식 1의 두 번째 항인 $C_{\mathrm{backend}}$를 어떻게 결정하느냐이다. 일반적인 아이디어는 서로 다른 백엔드에서 모든 연산자의 시간 비용을 합산한 다음, 비용이 가장 작은 백엔드를 선택하는 것이다.

$$
\begin{array}{cr}
C_{\mathrm{backend}} = \displaystyle\sum_{\mathrm{op}} C_{\mathrm{op}},
&
\qquad \text{(4)}
\end{array}
$$

앞에서와 마찬가지로 $C$는 비용을 나타내며, op는 연산자를 의미한다. 백엔드 비용 평가의 핵심은 $C_{\mathrm{op}}$ 항을 구하는 것이며, 이 값은 백엔드에 따라 달라진다. 관심 대상 GPU 백엔드(예: OpenCL/OpenGL/Vulkan)가 어떤 연산자를 지원하지 않으면 해당 연산자는 CPU에서 실행되도록 스케줄링된다. CPU 또는 GPU에서 실행되는 연산자의 비용은 다음과 같이 나타낼 수 있다.

$$
\begin{array}{cr}
C_{\mathrm{op}} =
\begin{cases}
\displaystyle
\frac{\mathrm{MUL}}{\mathrm{FLOPS}} \times 1000,
&
\text{if on CPU}, \\[6pt]
\displaystyle
\frac{\mathrm{MUL}}{\mathrm{FLOPS}} \times 1000 + t_{\mathrm{schedule}},
&
\text{if on GPU},
\end{cases}
&
\qquad \text{(5)}
\end{array}
$$

`MUL`은 곱셈 횟수를 뜻하며 연산자의 계산 복잡도를 나타낸다. `FLOPS`(FLoating-Operations Per Second)는 CPU 또는 GPU의 계산 능력을 정량화하는 일반적인 지표다. 위 식에서 볼 수 있듯이 백엔드로서 GPU가 CPU와 다른 주된 점은 추가 항 $t_{\mathrm{schedule}}$이 있다는 것이다. 이 항은 GPU를 위한 명령 버퍼와 명령 기술 정보를 준비하는 비용을 나타낸다. 특정 백엔드에서 `FLOPS`와 $t_{\mathrm{schedule}}$은 이미 알려진 상수다. 이 값을 구하는 자세한 방법은 부록을 참조하라.

표 1은 서로 다른 합성곱 설정에서 고정된 방식과 제안하는 방식 선택 기법의 성능을 비교한다. 슬라이딩 윈도와 Winograd는 각각 특정한 경우에는 특히 적합하지만 모든 경우에 보편적으로 우수하지는 않다. 제안하는 방법은 서로 다른 경우에 가장 우수하거나 최상에 가까운 결과를 얻으며, 이는 주장한 범용적 효율성을 부분적으로 보여준다.

![](./Table%2001.png)<br/>
**표 1.** 서로 다른 계산 방식의 추론 시간(ms) 비교. "Sliding"은 "sliding window"를 나타내고, "WinoMin/Max"는 "최소/최대 블록 크기를 사용하는 Winograd 합성곱"을 의미한다. 합성곱 인수 설정의 네 숫자는 차례대로 커널 크기, 입력 채널 수, 출력 채널 수, 입력의 공간 크기를 나타낸다. 가장 좋은 결과는 굵게 표시하고 두 번째로 좋은 결과는 밑줄로 표시했다.

**준비-실행 분리.** 프로그램을 실행하는 동안 계산은 일반적으로 메모리 할당 및 해제와 뒤섞여 이루어진다. 그러나 모바일 애플리케이션에서는 메모리 관리에 소요되는 시간을 무시할 수 없다. 입력 크기는 정해져 있거나 목표 크기로 전처리할 수 있으므로, MNN은 모든 연산을 가상으로 순회하면서 각 메모리 할당과 해제를 합산하여 전체 그래프에 정확히 필요한 메모리 양을 추론할 수 있다. 이에 따라 MNN은 사전 추론 단계에서 필요한 메모리를 메모리 풀로 미리 할당하고, 이후의 추론 세션에서 이를 재사용한다. 전체 과정을 그림 3에 나타냈다.

![](./Figure%2003.png)<br/>
**그림 3.** MNN의 메모리 최적화: 메모리 할당을 계산에서 분리한다.

준비와 실행을 분리하면 선택한 방식이 GPU와 관련된 경우에도 MNN이 더 나은 성능을 얻는다는 점에 주목할 필요가 있다. 명령 버퍼와 관련 명령 기술 정보를 설정하는 데 어느 정도 시간이 들며, 이 작업이 추론 성능에 부정적인 영향을 주기 때문이다.

이 단순한 아이디어는 실제로 매우 효과적일 수 있다. 이 기법을 사용하면 추론 시간이 CPU에서 약 7%∼8%, GPU에서 50%∼75% 감소할 수 있다. 자세한 결과는 표 2에 제시한다.

![](./Table%2002.png)<br/>
**표 2.** 제안하는 준비-실행 분리를 적용하지 않은 경우(w/o)와 적용한 경우(w/)의 추론 시간(ms) 비교. 하드웨어 설정: (1) MI6 – CPU: Kryo 280, GPU: Adreno 540, (2) P10 – CPU: Cortex A73, GPU: Mali G71.

### 3.3. 커널 최적화

커널은 연산자의 구체적인 구현이며, 커널 최적화는 알고리즘과 스케줄이라는 두 측면으로 나눌 수 있다(Jiang et al., 2018). 다시 말해 산술 복잡도가 가장 낮은 최적 알고리즘을 선택하고, 사용할 수 있는 하드웨어 자원을 최대한 활용하여 가능한 한 빠르게 실행해야 한다.

#### 3.3.1. Winograd 최적화

Winograd 알고리즘은 Shmuel Winograd가 제안한 잘 알려진 최소 필터링 알고리즘(Winograd, 1980)으로, DNN의 합성곱을 가속하는 데 사용되어 왔다(Lavin & Gray, 2016).

크기가 $[i_w, i_h, i_c]$인 입력 특징 맵과 크기가 $[o_w, o_h, o_c]$인 출력 특징 맵이 주어졌을 때, Winograd 합성곱은 다음과 같이 나타낼 수 있다.

$$
\begin{array}{cr}
Y = A^T
\left[
\displaystyle\sum_{\mathrm{channel}}
\left(GWG^T\right)
\odot
\left(B^T X B\right)
\right]
A,
&
\qquad \text{(6)}
\end{array}
$$

여기서 $G$, $B$, $A$는 각각 커널 $W$(공간 크기 $[k, k]$), 입력 $X$(공간 크기 $[n + k−1, n + k−1]$), 출력 $Y$(공간 크기 $[n, n]$)를 위한 변환 행렬이다. 이 세 행렬은 $W$와 $X$의 형태에만 의존한다.

제안하는 Winograd 생성기를 기반으로 파이프라이닝과 SIMD 같은 널리 쓰이는 병렬 컴퓨팅 기법을 함께 적용하여 Winograd 합성곱을 최적화한다.

**(1) 블록 분할 및 파이프라이닝.** Winograd 합성곱(Lavin & Gray, 2016)에서 $X$는 전체 입력 특징 맵이 아니라 작은 타일이다. 따라서 첫 번째 문제는 블록 분할, 즉 $n$을 어떻게 결정하느냐이다.

이 문제를 해결하기 위해 *출력* 관점에서 블록을 분할한다. 크기가 $[o_w, o_h, o_c]$인 출력에서 $T$를 병렬 계산의 배수, 즉 한 번에 계산하는 출력 블록의 수라고 하자. 그러면 다음 관계가 성립한다.

$$
\begin{array}{cr}
T =
\left\lfloor
\frac{o_w o_h}{\hat{n}^2}
\right\rfloor,
&
\qquad \text{(7)}
\end{array}
$$

여기서 $\hat{n}$은 앞서 설명한 대로 사전 추론 단계에서 결정한 최적 출력 타일 크기다(식 2).

여러 블록을 함께 계산할 때는 지연 시간을 감추기 위해 파이프라인 정지를 최대한 피해야 한다. 널리 알려진 방법은 파이프라인의 데이터 의존성을 피하는 것이다(Kennedy & Allen, 2001). MNN에서는 세심한 *어셈블리 명령어 재배치*를 통해 이를 달성한다.

**(2) 아다마르 곱 최적화.** 아다마르 곱은 Winograd 합성곱에서 필수적인 단계다(식 6 참조). 그러나 메모리 접근에 많은 시간이 소요되어 전체 가속 효과를 떨어뜨리는 문제가 있다.

식 6을 보면 합산과 아다마르 곱을 내적으로 변환할 수 있음을 알 수 있다. 여러 내적을 결합하면 행렬 곱셈이 되며, 행렬 곱셈은 병렬성을 활용하고 메모리 접근 오버헤드를 분산하기에 좋은 형태다. 이러한 관점에서 *데이터 레이아웃 재배치*를 바탕으로 아다마르 곱을 행렬 곱셈으로 변환하는 방법을 제안한다. 그 결과 얻는 새로운 데이터 레이아웃을 NC4HW4라고 한다(DMLC, 2016; Apple, 2014). 간단히 말해 NC4HW4는 $V$개의 데이터 원소(본 논문에서는 $V = 4$)를 하나의 단위로 분리하여 텐서에 새로운 차원을 만드는 데이터 레이아웃 재배치 방식이다. $V$개 원소를 메모리에 연속으로 배치함으로써 CPU의 벡터 레지스터를 활용하여 단일 명령어, 즉 SIMD로 $V$개의 데이터를 계산할 수 있다. 이러한 재배치 이후의 Winograd 합성곱을 그림 4에 나타냈다.

![](./Figure%2004.png)<br/>
**그림 4.** MNN에서 최적화한 Winograd 알고리즘의 도식(*컬러로 보는 것을 권장*).

**(3) Winograd 생성기.** Winograd를 사용하는 기존 추론 프레임워크 대부분(Google, 2017a; Tencent, 2017; Xiaomi, 2018)은 흔히 쓰이는 커널 및 입력 크기에 대한 $A$, $B$, $G$ 행렬을 소스 코드에 하드코딩한다(Google; Xiaomi). 이 방식은 새로운 경우에 대응하는 확장성이 비교적 낮다. 반면 MNN은 제안하는 *Winograd 생성기*를 통해 범용 인터페이스를 유지하며, *임의의* 커널 크기와 입력 크기에 대한 Winograd 합성곱을 지원한다. $A$와 $B$ 행렬을 생성하기 위해 다음 공식을 사용한다.

$$
\begin{array}{cr}
\begin{aligned}
&x \cdot (x-f)(x+f) \cdot (x-2f)(x+2f) \cdots \\
&\left(x-\frac{(n+k-1)f}{2}\right)
 \left(x+\frac{(n+k-1)f}{2}\right),
\end{aligned}
&
\qquad \text{(8)}
\end{array}
$$

여기서 $f$는 수치 오차를 최소화하는 데 사용하는 스칼라다. 본 논문에서는 $f = 0.5$로 설정한다.

#### 3.3.2. 대규모 행렬 곱셈 최적화

앞서 설명했듯이(3.2절), MNN에서는 커널 크기가 1인 합성곱 연산을 대규모 행렬 곱셈으로 변환하고 Strassen 알고리즘(Strassen, 1969)으로 가속한다. 저자들이 아는 한 MNN은 대규모 행렬 곱셈을 가속하기 위해 Strassen 알고리즘을 채택한 *최초의* 모바일 추론 엔진이다.

Strassen은 비용이 큰 곱셈을 비용이 작은 덧셈으로 대체하는 빠른 알고리즘이며, *재귀적으로* 적용할 때 가속 효과가 극대화된다(Blahut, 2010). 실제로는 재귀를 언제 중단할지 결정해야 한다. 현대 프로세서에서는 곱셈과 덧셈의 비용이 거의 같으므로 연산 횟수를 통해 두 비용을 비교할 수 있다. MNN에서 크기 $[n, k] × [k, m] ⇒ [n, m]$의 행렬 곱셈을 직접 수행하면 곱셈이 $mnk$회 필요하지만, Strassen을 사용하면 $7 \cdot \frac{m}{2}\frac{n}{2}\frac{k}{2}$회의 곱셈만 필요하다. Strassen을 사용할 때 추가되는 비용은 크기가 $\left[\frac{m}{2}, \frac{k}{2}\right]$인 행렬 덧셈 4회, 크기가 $\left[\frac{n}{2}, \frac{k}{2}\right]$인 행렬 덧셈 4회, 크기가 $\left[\frac{m}{2}, \frac{n}{2}\right]$인 행렬 덧셈 7회다. 따라서 이득이 비용보다 큰 경우에만 재귀를 계속하며, 이를 다음과 같이 나타낼 수 있다.

$$
\begin{array}{cr}
mnk - 7 \cdot \frac{m}{2}\frac{n}{2}\frac{k}{2}
>
4 \cdot \frac{m}{2}\frac{k}{2}
+
4 \cdot \frac{n}{2}\frac{k}{2}
+
7 \cdot \frac{m}{2}\frac{n}{2}.
&
\qquad \text{(9)}
\end{array}
$$

이 부등식이 더 이상 성립하지 않으면 Strassen 재귀를 중단해야 한다.

표 3은 서로 다른 행렬 크기에서 직접 행렬 곱셈과 비교한 Strassen의 이점을 보여준다. Strassen 방식은 직접 계산 방식보다 7.5%∼13.5% 더 나은 성능을 보인다.

![](./Table%2003.png)<br/>
**표 3.** P10에서 최적화한 Strassen 알고리즘을 사용하지 않은 경우(w/o)와 사용한 경우(w/)의 행렬 곱셈 소요 시간(ms) 비교. 행렬 크기 $(a, b, c)$는 크기 $[a, b]$인 행렬과 크기 $[b, c]$인 행렬의 곱셈을 의미한다.

### 3.4. 백엔드 추상화

백엔드 추상화 모듈은 모든 하드웨어 플랫폼(예: GPU, CPU, TPU)과 소프트웨어 솔루션(예: OpenCL, OpenGL, Vulkan)을 통일된 `Backend` 클래스로 캡슐화하기 위해 도입되었다. `Backend` 클래스를 통해 자원 관리, 메모리 할당, 스케줄링을 구체적인 연산자 구현과 분리한다.

그림 5에 나타낸 것처럼 `Backend` 클래스는 여러 추상 함수로 구성된다. 메모리 관리 측면에서 `onAcquireBuffer`는 텐서를 위한 새 메모리를 할당하고, `onReleaseBuffer`는 이를 해제한다. 연산자 구현 측면에서 `onCreate`는 각 연산자의 실행 인스턴스를 생성하도록 설계되었다.

![](./Figure%2005.png)<br/>
**그림 5.** MNN의 `Backend` 클래스(*컬러로 보는 것을 권장*).

이 모듈의 장점은 세 가지다.

**(1) 복잡성 감소.** 수많은 연산자와 헤아릴 수 없이 많은 장치 때문에 연산자 최적화는 결코 단순한 작업이 아니다. 주된 과제는 이기종 백엔드마다 메모리 할당·해제와 데이터 스케줄링 등 자원을 관리하는 방식이 일반적으로 서로 다르다는 데 있다. 이러한 문제를 직접 처리하면 구현에서 오류가 발생하기 쉽고 작업도 번거로워진다. `Backend` 클래스는 GPU 셰이더 같은 자원의 로딩을 통일된 방식으로 관리하고, 연산자의 선언에 따라 최적의 메모리 할당을 수행한다. 백엔드 추상화를 통해 MNN은 작업을 서로 독립적인 두 부분으로 나눈다. "프런트엔드 연산자" 개발자는 빠른 연산자 실행을 위한 효율적인 구현에 집중할 수 있으며, 관련 없는 백엔드 세부 사항은 모두 숨겨진다. "백엔드" 개발자는 서로 다른 백엔드 사양을 활용하고 더 편리한 API를 제공하는 데 전념할 수 있다. 오픈 소스 프로젝트에서는 기여 진입 장벽을 낮추는 일이 매우 중요하므로, 이러한 역할 분리는 실무적으로 큰 의미가 있다.

**(2) 하이브리드 스케줄링 지원.** 이기종 컴퓨팅은 주로 백엔드 선택과 서로 다른 백엔드 사이의 데이터 전송을 포함한다. MNN에서 추론 세션을 생성할 때 대상 백엔드를 구성할 수 있다. 여러 백엔드를 사용할 수 있으면 MNN은 앞서 설명한 백엔드 평가(3.2절)에 따라 각 연산자에 가장 적합한 백엔드를 결정한다. 그 결과 MNN은 한 번의 추론 안에서도 서로 다른 백엔드에서 연산자를 실행하는 방식을 유연하게 조합할 수 있다. 예를 들어 합성곱은 CPU에서 실행하고, 뒤따르는 ReLU 활성화는 GPU에서 실행할 수 있다. `Backend` 클래스 덕분에 개발자는 스케줄링과 데이터 전송을 걱정할 필요가 없으며, 이 작업은 내부에서 자동으로 진행된다.

**(3) 더 가벼운 구조.** 각 백엔드 구현은 MNN의 통일된 인터페이스를 유지하면서 독립적인 구성 요소로 동작할 수 있다. 특정 장치에서 어떤 백엔드를 사용할 수 없다면 해당 구체 구현을 전체 프레임워크에서 쉽게 떼어낼 수 있다. 예를 들어 Metal(Apple, 2014)은 Android 플랫폼에서 지원되지 않으므로, 구체적인 연산자 구현을 건드리지 않고 Metal 모듈만 제거할 수 있다. 연산자와 백엔드를 분리함으로써 MNN은 경쟁력 있는 경량성을 확보한다. 모바일 애플리케이션에는 바이너리 크기에 엄격한 제약이 있으므로 이는 매우 중요하다.

저자들이 아는 한 MNN은 이러한 통일된 백엔드 인터페이스를 통해 가장 포괄적인 백엔드를 지원하는 추론 엔진이다(표 4 참조). 또한 사용자가 NPU, FPGA 같은 새로운 백엔드를 통합할 수 있을 만큼 확장성이 높다.

![](./Table%2004.png)<br/>
**표 4.** 여러 모바일 추론 엔진의 백엔드 비교. "–"는 논문 제출 당시 해당 모바일 엔진이 그 종류의 백엔드를 지원하지 않음을 뜻하고, "/"는 "해당 없음"을 나타낸다. Metal(Apple, 2014)은 iOS에서 사용하는 Apple 전용 GPU 표준이며, OpenCL(Khronos, 2009), OpenGL(Khronos, 1992), Vulkan(Khronos, 2015)은 Android에서 함께 사용되는 표준이다.

### 3.5. 방법 요약

이 절에서는 그림 6에 제시한 다른 대표적인 설계 패러다임과 비교하면서 MNN의 설계 철학을 설명하고, MNN에 고유한 몇 가지 장점을 강조한다.

![](./Figure%2006.png)<br/>
**그림 6.** MNN과 널리 사용되는 다른 두 종류의 모바일 추론 엔진 사이의 설계 패러다임 비교.

추론 엔진에서 높은 성능은 개발자의 채택 여부를 결정하는 핵심 요소다. 이러한 동기에서 여러 추론 엔진(예: TF-Lite, NCNN, TVM)은 서로 다른 방식으로 더 나은 성능을 달성하기 위해 지속적으로 최적화하고 있다.

MNN은 높은 성능이라는 기본 요구 사항을 충족하기 위해 많은 부분을 개선했다. 압도적으로 다양한 연산자를 고려할 때 사례별 최적화만으로는 충분하지 않다. 이 방식은 상당히 단순하고 효과적일 수 있지만, 일부 연산자가 최적화에서 제외되어 성능 병목이 되는 문제가 자주 발생한다(4.2절의 예 참조). 대신 MNN은 먼저 *더 작은 세분성*의 계산 집약적 단위, 즉 기본 행렬 곱셈을 찾아내고 빠른 알고리즘과 병렬화 기법으로 이를 고도로 최적화한다. 그 결과 이 기본 단위를 바탕으로 구현된 연산자는 별도의 특별한 최적화 없이도 자연스럽게 가속의 이점을 얻는다.

또한 유지보수성, 확장성, 배포 비용은 추론 엔진의 장기적인 성장에 모두 큰 영향을 준다. TVM의 자동 튜닝과 비교할 때 MNN은 더 짧은 시간에 최적 계산 방식을 선택할 수 있고, 제안하는 사전 추론 메커니즘을 통해 런타임 최적화를 실현한다. 탐색 단계를 오프라인 컴파일에서 온라인 사전 추론으로 옮김으로써 iOS 코드 서명 같은 바이너리 검증의 제약도 피할 수 있다는 점에 주목하라. 이와 동시에 MNN은 백엔드의 내부 세부 사항을 숨기는 통일된 인터페이스 집합을 제공하므로 자연스럽게 모듈성을 갖춘다. 이 속성은 MNN을 경량으로 만들 뿐 아니라 기여자가 더 많은 백엔드로 MNN을 확장하기도 쉽게 해준다. 이는 서로 다른 백엔드가 지원하는 연산자 수를 제시한 표 4에서도 부분적으로 확인할 수 있다.

## 4. 벤치마크 실험

이 절에서는 MNN의 성능을 종합적으로 평가한다. 먼저 실험 설정을 설명한 다음, 여러 하드웨어 플랫폼과 신경망에서 다른 모바일 추론 엔진과 비교한 실험 결과를 제시한다. 마지막으로 MNN이 프로덕션 환경에서 적용된 실제 온라인 사례를 소개한다.

### 4.1. 실험 설정

* **추론 엔진.** CoreML(Apple, 2017), TF-Lite(Google, 2017a), NCNN(Tencent, 2017), MACE(Xiaomi, 2018)를 포함한 최신 모바일 추론 엔진과 성능을 비교한다.

* **장치.** iOS에서는 벤치마크에 널리 사용되는<sup>1</sup> iPhone 8과 iPhone X(프로세서: Apple A11 Bionic)를 사용한다. Android에서는 MI6(프로세서: Snapdragon 835)와 Mate20(프로세서: Kirin 980)을 사용한다.

  <sup>1</sup> https://www.tensorflow.org/lite/performance/benchmarks

* **CPU와 GPU.** (1) CPU의 경우 현대 장치는 일반적으로 2개 또는 4개의 처리 코어를 갖추고 있으며 멀티스레딩이 일반적인 가속 기법이라는 점을 고려하여 2개 스레드와 4개 스레드를 평가한다. CPU 친화도는 NCNN 벤치마크(Tencent, 2017)와 마찬가지로 사용 가능한 모든 코어를 사용하도록 설정한다. (2) GPU의 경우 iPhone에서 Metal Performance Shaders를 평가한다. Android 장치에서는 MNN이 세 백엔드를 모두 지원하므로(표 4 참조) 세 가지 표준 백엔드인 OpenCL, OpenGL, Vulkan을 평가한다.

* **신경망.** 모바일 애플리케이션에서 폭넓게 사용되어 온 MobileNet-v1(Howard et al., 2017), SqueezeNet-v1.1(Iandola et al., 2016), ResNet-18(He et al., 2016)을 벤치마크 신경망으로 선택한다.

* **실행 설정.** 224 × 224 RGB 이미지 한 장, 즉 배치 크기 1에 대한 추론 시간을 10회 실행의 평균으로 보고한다. 다른 연구(Tencent, 2017; Google, 2017a)와 공정하게 비교하기 위해 벤치마크 전에 워밍업 추론을 한 차례 수행한다.

### 4.2. 실험 결과

**서로 다른 스마트폰과 신경망에서의 성능.** 그림 7은 MNN과 다른 네 추론 엔진의 성능을 비교한다. 이를 통해 다음과 같은 사실을 관찰할 수 있다.

![](./Figure%2007.png)<br/>
**그림 7.** MobileNet-v1(왼쪽), SqueezeNet-v1.1(가운데), ResNet-18(오른쪽)의 추론 시간(ms) 비교. 첫 번째 행: 2개 스레드를 사용한 CPU. 두 번째 행: 4개 스레드를 사용한 CPU. 세 번째 행: GPU. (*컬러로 보는 것을 권장*)

(1) 전반적으로 MNN은 스마트폰, 백엔드, 신경망과 관계없이 *거의 모든* 설정에서 다른 추론 엔진보다 약 20%∼40% 높은 성능을 보인다.

(2) CPU의 경우 MNN의 4스레드 추론은 평균적으로 iOS 플랫폼에서 다른 엔진보다 약 30% 빠르고, Android 플랫폼(예: Mate20)에서는 약 34% 빠르다.

(3) iPhone의 Metal GPU 백엔드에서 MNN은 TF-Lite보다 훨씬 빠르며, CoreML보다는 약간 느리지만 여전히 대등한 수준이다. CoreML은 iOS에 맞춰 설계된 Apple 전용 GPU 솔루션인 반면 MNN은 여러 운영체제의 백엔드를 지원하도록 설계되었으므로 이는 합리적인 결과다. Android GPU 백엔드에서는 다른 엔진에 성능상의 사각지대가 나타나는 경우가 많다. 예를 들어 Vulkan 백엔드를 사용하는 NCNN은 MI6에서 그다지 빠르지 않고, OpenGL을 사용하는 TF-Lite는 ResNet-18 신경망에서 여전히 개선의 여지가 크다. 반면 MNN은 *모든* 하드웨어 플랫폼과 신경망에서 우수한 결과를 얻는다. 이러한 포괄적인 성능 우위는 사례별로 무거운 최적화를 적용한 결과가 아니라 제안하는 반자동 탐색 아키텍처를 통해 달성한 것임에 주목하라.

(4) 고사양 장치(예: iPhone 8과 iPhone X)에서 MNN을 사용하는 멀티스레드 CPU 추론은 GPU 백엔드를 사용하는 추론과 비교해도 매우 경쟁력이 있다. 이는 MNN이 제안하는 심층적인 커널 최적화가 효과적임을 보여준다.

**사례별 최적화의 병목.** 그림 8은 Inception-v3(Szegedy et al., 2015)에서 사례별 최적화가 낮은 성능을 보이는 예를 나타낸다. NCNN의 추론 시간이 다른 엔진보다 비정상적으로 훨씬 길다는 사실을 분명히 확인할 수 있다. 이는 이 신경망의 일부 특수 연산자, 예를 들어 1 × 7 및 7 × 1 합성곱이 당시 NCNN에서 최적화되어 있지 않기 때문이다. 그 결과 이 연산자들이 실행 중 병목이 되어 전체 성능을 심각하게 떨어뜨린다. 이 사례는 사례별 최적화의 확장성이 제한적임을 보여준다. MNN의 계산 방식은 여러 합성곱 사례에 적용할 수 있는 일반적인 방식이므로 이러한 문제에서 자유롭다.

![](./Figure%2008.png)<br/>
**그림 8.** Huawei P20(Kirin 970)에서 평가한 Inception-v3의 사례별 최적화 병목. "MNN-Vul"은 "MNN-Vulkan"을 의미하며, "MACE-CL"은 "MACE-OpenCL"을 나타낸다.

**TVM과의 비교.** 그림 9와 같이 여러 신경망에서 MNN과 TVM의 성능을 비교한다. MNN은 모델별 튜닝과 최적화를 적용하지 않음에도 고무적인 성능을 보이며 TVM보다 약간 더 빠르기까지 하다. 또한 TVM에서 자동 튜닝을 사용해 모델별 코드를 생성하고 컴파일하면 일반적으로 어느 정도의 배포 오버헤드가 발생한다. 표 5<sup>2</sup>에는 TVM으로 ResNet-18을 자동 튜닝하고 컴파일하는 데 소요되는 시간을 제시한다. 단일 장치에서 적은 횟수로만 튜닝을 시도해도 TVM은 코드를 생성하는 데 많은 시간이 걸린다. 대부분의 모바일 애플리케이션은 매우 다양한 장치 유형을 지원하므로 코드 생성 과정에는 훨씬 더 많은 시간이 들고 서버 같은 자원도 더 많이 필요하다. 이는 많은 개발자가 감당하기 어렵다. MNN에서는 모든 최적화를 성능 손실 없이 런타임에 수행하므로 이러한 문제가 없다.

<sup>2</sup> 컴파일 결과가 자동 튜닝보다 빠른 이유가 궁금할 수 있다. TVM은 개발자가 자동 튜닝된 코드를 수작업으로 최적화한 구현으로 교체할 수 있는 메커니즘, 즉 `tensorize`를 제공한다. 따라서 저자들이 최적화한 직접 컴파일 방식이 자동 튜닝보다 더 나은 성능을 보이는 것은 합리적이다.

![](./Figure%2009.png)<br/>
**그림 9.** Huawei P20 Pro(SoC: HiSilicon Kirin 970)에서 MNN과 TVM(DMLC, 2016)의 CPU 추론 시간(ms) 비교. "Mo/Sq/Res/Inc"는 각각 MobileNet/SqueezeNet/ResNet/Inception의 약어다. TVM 데이터는 공개된 벤치마크에서 가져왔다: https://github.com/dmlc/tvm/wiki/Benchmark.

![](./Table%2005.png)<br/>
**표 5.** Samsung Galaxy S8(GPU: Adreno 540)에서 TVM으로 ResNet-18을 자동 튜닝하고 컴파일하는 데 소요되는 시간(s).

### 4.3. 온라인 적용 사례

상품 검색은 전자상거래 플랫폼에서 쇼핑할 때 반드시 거치는 단계다. 그러나 상품 범주의 복잡성 때문에 전통적인 텍스트 검색 방식만으로는 오늘날 사용자의 기대를 충족하기 어렵다. 따라서 이미지로 상품을 검색하는 기능은 전자상거래 플랫폼의 필수 기능이 되었다. 이 절에서는 이러한 시나리오에 MNN을 적용한 실제 사례를 소개한다.

해당 전자상거래 애플리케이션에서는 모바일 장치에서 MNN으로 딥러닝 모델을 실행하여 주요 객체를 검출하고, 그 검출 결과를 상품 검색에 사용한다. 이 서비스는 500종이 넘는 모바일 장치를 지원하며 일일 사용자가 1,000만 명을 넘는다. 표 6은 이 서비스에서 가장 많이 사용되는 상위 5개 장치와 평균 추론 시간을 보여준다. MNN은 장치가 매우 다양함에도 모든 장치에서 평균 90.2ms의 추론 시간을 달성하여 안정적이고 매끄러운 검색 경험을 제공한다. 이는 MNN의 뛰어난 범용성을 보여준다.

![](./Table%2006.png)<br/>
**표 6.** 매우 큰 규모의 실제 프로덕션 사례에서 MNN을 사용하는 상위 5개 인기 장치와 평균 추론 시간(AIT, ms).

## 5. 결론 및 향후 연구

모바일 추론 엔진은 딥러닝 모델을 모바일 애플리케이션에 배포하는 데 핵심적인 역할을 한다. 모델 호환성과 장치 다양성이라는 과제를 해결하기 위해 범용성과 효율성을 모두 최대한 확보하는 새로운 모바일 엔진 설계 패러다임인 반자동 탐색을 제안하는 Mobile Neural Network (MNN)를 소개했다. 철저한 커널 최적화 및 백엔드 추상화와 결합한 사전 추론 메커니즘은 MNN에 우수한 범용성과 최신 수준의 온디바이스 추론 성능을 제공한다.

MNN은 여전히 빠르게 발전하고 있으며 여러 측면에서 개선되고 있다. 예를 들면 (1) 백엔드 평가 중 자동 튜닝 적용, (2) 가지치기 같은 모델 압축 도구를 통합하여 즉석에서 모델 경량화, (3) 사용 편의를 위한 더 많은 도구 제공, (4) JavaScript와 Python을 포함한 더 많은 언어 지원 등이 있다.

## 감사의 말

유익한 논의를 나눈 Chaoyue Niu와 본 연구를 개선하는 데 귀중한 의견을 제공한 익명의 심사위원들에게 감사드린다.

## 참고문헌

Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., Devin, M., Ghemawat, S., Irving, G., Isard, M., et al. TensorFlow: A system for large-scale machine learning. In IEEE Symposium on Operating Systems Design and Implementation, 2016.

Apple. Metal: Accelerating graphics and much more. https://developer.apple.com/metal/, 2014. Accessed: 2019-09-01.

Apple. CoreML. https://developer.apple.com/documentation/coreml, 2017. Accessed: 2019-09-01.

Ashari, A., Tatikonda, S., Boehm, M., Reinwald, B., Campbell, K., Keenleyside, J., and Sadayappan, P. On optimizing machine learning workloads via kernel fusion. In ACM SIGPLAN Notices, 2015.

Baidu. Anakin. https://github.com/PaddlePaddle/Anakin, 2018. Accessed: 2019-09-01.

Bastien, F., Lamblin, P., Pascanu, R., Bergstra, J., Goodfellow, I., Bergeron, A., Bouchard, N., Warde-Farley, D., and Bengio, Y. Theano: new features and speed improvements. In NeurIPS Workshop, 2012.

Blahut, R. E. Fast algorithms for signal processing. Cambridge University Press, 2010.

Chen, T., Li, M., Li, Y., Lin, M., Wang, N., Wang, M., Xiao, T., Xu, B., Zhang, C., and Zhang, Z. Mxnet: A flexible and efficient machine learning library for heterogeneous distributed systems. arXiv preprint arXiv:1512.01274, 2015.

DMLC. TVM: Tensor virtual machine, open deep learning compiler stack. https://github.com/dmlc/tvm, 2016. Accessed: 2019-09-01.

Google. TensorFlow Winograd. https://github.com/tensorflow/tensorflow/blob/9590c4c32dd4346ea5c35673336f5912c6072bf2/tensorflow/core/kernels/winograd_transform.h. Accessed: 2019-09-01.

Google. Neural Networks API. https://developer.android.google.cn/ndk/guides/neuralnetworks, 2016. Accessed: 2019-09-01.

Google. TensorFlow Lite. https://tensorflow.google.cn/lite, 2017a. Accessed: 2019-09-01.

Google. XLA: Accelerated linear algebra. https://developers.googleblog.com/2017/03/xla-tensorflow-compiled.html, 2017b. Accessed: 2019-09-01.

Guo, L., Hua, L., Jia, R., Zhao, B., Wang, X., and Cui, B. Buying or browsing?: Predicting real-time purchasing intent using attention-based deep network with multiple behavior. In SIGKDD, 2019.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In CVPR, 2016.

Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., and Adam, H. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017.

Iandola, F., Moskewicz, M., and Ashraf, K. SqueezeNet: Alexnet-level accuracy with 50x fewer parameters and <0.5MB model size. arXiv preprint arXiv:1602.07360, 2016.

Jia, Y., Shelhamer, E., Donahue, J., Karayev, S., Long, J., Girshick, R., Guadarrama, S., and Darrell, T. Caffe: Convolutional architecture for fast feature embedding. In ACM MM, 2014.

Jiang, Z., Chen, T., and Li, M. Efficient deep learning inference on edge devices. In MLSys, 2018.

Kennedy, K. and Allen, J. R. Optimizing compilers for modern architectures: a dependence-based approach. Morgan Kaufmann, 2001.

Khronos, G. OpenGL: Open Graphics Library. https://opengl.org/, 1992. Accessed: 2019-09-01.

Khronos, G. OpenCL: Open Computing Language. https://www.khronos.org/opencl/, 2009. Accessed: 2019-09-01.

Khronos, G. Vulkan. https://www.khronos.org/vulkan, 2015. Accessed: 2019-09-01.

Lavin, A. and Gray, S. Fast algorithms for convolutional neural networks. In CVPR, 2016.

LeCun, Y., Bengio, Y., and Hinton, G. Deep learning. Nature, 521(7553):436, 2015.

Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z., Lin, Z., Desmaison, A., Antiga, L., and Lerer, A. Automatic differentiation in pytorch. In NeurIPS Workshop, 2017.

Rastegari, M., Ordonez, V., Redmon, J., and Farhadi, A. Xnor-net: Imagenet classification using binary convolutional neural networks. In ECCV, 2016.

Reddi, V. J., Cheng, C., Kanter, D., Mattson, P., Schmuelling, G., Wu, C.-J., Anderson, B., Breughe, M., Charlebois, M., Chou, W., et al. Mlperf inference benchmark. arXiv preprint arXiv:1911.02549, 2019.

Shi, W., Cao, J., Zhang, Q., Li, Y., and Xu, L. Edge computing: Vision and challenges. IEEE Internet of Things Journal, 3(5):637–646, 2016.

Strassen, V. Gaussian elimination is not optimal. Numerical Mathematics, 13(4):354–356, 1969.

Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., and Rabinovich, A. Going deeper with convolutions. In CVPR, 2015.

Tencent. NCNN. https://github.com/Tencent/ncnn, 2017. Accessed: 2019-09-01.

Wei, R., Schwartz, L., and Adve, V. DLVM: A modern compiler infrastructure for deep learning systems. arXiv preprint arXiv:1711.03016, 2017.

Winograd, S. Arithmetic complexity of computations, volume 33. SIAM, 1980.

Xiaomi. MACE Winograd. https://github.com/XiaoMi/mace/blob/9b0b03c99cf73cd019050c6b9ee80a4753265da0/mace/ops/arm/fp32/conv_2d_3x3_winograd.cc. Accessed: 2019-09-01.

Xiaomi. MACE: Mobile ai compute engine. https://github.com/XiaoMi/mace, 2018. Accessed: 2019-09-01.

Yu, D., Eversole, A., Seltzer, M., Yao, K., Huang, Z., Guenter, B., Kuchaiev, O., Zhang, Y., Seide, F., Wang, H., et al. An introduction to computational networks and the computational network toolkit. Microsoft Technical Report MSR-TR-2014–112, 2014.

## A. MLPerf 평가

Pixel 3의 CPU 스레드 4개에서 벤치마크 도구 MLPerf(Reddi et al., 2019)를 사용하여 MNN의 MobileNet-v2 벤치마크도 수행했다. 결과는 표 7에 제시한다.

![](./Table%2007.png)<br/>
**표 7.** MLPerf 벤치마크 결과.

## B. Pixel 휴대전화에서의 추가 비교

TF-Lite와 더 자세히 비교하기 위해 Pixel 2와 Pixel 3의 CPU에서 Inception-v3 부동소수점 모델도 평가했다. 표 8에 결과를 제시한다. 본 논문의 주요 결과와 마찬가지로, MNN은 단일 스레드와 멀티스레드 모두에서 일관되게 TF-Lite보다 빠르다.

![](./Table%2008.png)<br/>
**표 8.** Pixel 휴대전화에서의 CPU 추론 시간(ms) 비교.

## C. 백엔드 비용 평가

CPU와 GPU는 모두 프로세서의 성능을 측정하기 위해 `FLOPS`를 사용한다. $t_{\mathrm{schedule}}$ 항은 GPU에만 있다. 각 값은 다음과 같이 결정한다.

* **`FLOPS`.** CPU의 경우 운영체제가 Linux 또는 Android이면 각 CPU 코어의 최대 주파수에 접근할 수 있다. 그중 가장 높은 $k$개의 주파수를 선택하여 더한 값을 `FLOPS` 항으로 사용한다. 여기서 $k$는 2개 또는 4개처럼 미리 지정한 스레드 수다. 그 밖의 CPU 시스템에서는 `FLOPS`를 2 × 10<sup>9</sup>로 설정한다. GPU의 경우 실제 실행을 통해 `FLOPS`를 추정한다. 구체적으로 MobileNet-v1 신경망을 100회 실행하여 널리 사용되는 여러 모바일 GPU의 `FLOPS` 값을 얻는다. 그 결과를 아래 목록에 제시한다. 이 목록에 없는 GPU의 `FLOPS`는 일반적인 경우처럼 CPU보다 빠르다는 가정 아래 4 × 10<sup>9</sup>로 설정한다.

  GPU `FLOPS` 목록(10<sup>9</sup>): Mali-T860: 6.83; Mali-T880: 6.83; Mali-G51: 6.83; Mali-G52: 6.83; Mali-G71: 31.61; Mali-G72: 31.61; Mali-G76: 31.61; Adreno (TM) 505: 3.19; Adreno (TM) 506: 4.74; Adreno (TM) 512: 14.23; Adreno (TM) 530: 25.40; Adreno (TM) 540: 42.74; Adreno (TM) 615: 16.77; Adreno (TM) 616: 18.77; Adreno (TM) 618: 18.77; Adreno (TM) 630: 42.74; Adreno (TM) 640: 42.74.

* **$t_{\mathrm{schedule}}$.** 이 항은 사용하는 그래픽 API에 따라 달라진다. OpenCL과 OpenGL에서는 `clEnqueueNDRKernel` 같은 API를 호출하는 일반적인 평균 시간에 근거하여 경험적으로 0.05ms로 설정한다. Vulkan은 상대적으로 시간이 적게 드는 `commandBuffer` 제출만 필요하므로 $t_{\mathrm{schedule}}$을 0.01ms로 추정할 수 있다.
