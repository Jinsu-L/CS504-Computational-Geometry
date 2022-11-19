import numpy as np
import time


class Evaluation:
    def __init__(self) -> None:
        pass


    def evaluate(self, model, dataset, experiment_params : dict = None):
        """
        특정 알고리즘이랑 데이터셋이 주어지면 시간을 체크해서 리턴
        데이터셋을 dimension, size 별로
        random shuffle 후 체크를 여러번 해서? 평균과 분산을 체크하는 것이 필요할까?

        return은
        
        기본은 복원 추출로 진행하자!
        
        input: 데이터셋하고, 실험 데이터 사이즈 리스트, None 일 경우 전체를 해서 테스트, Shuffle 옵션

        데이터셋 정보 | 실험번호 | 데이터사이즈 | 시간 (ms)


        """

        result = []
        for experiment_ops in experiment_params:
            # 실험 dataset 생성
            test_datasets = dataset.get_data(experiment_ops)

            # N 번 실행 해서 평균과 분산 계산용
            for experiment_no, expset in enumerate(test_datasets):
                data_size = len(expset)

                # 실험 진행
                start_time = time.time()
                result = model.eval(test_dataset)
                iter_result.append(start_time - tiem.time())

        
        return result

