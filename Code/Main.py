import csv
from itertools import combinations


def main():
    
    data = load_dataset_from_csv("./Data/supermarket.csv")
    
    if data is None:
        return
    
    min_support = 0.3
    min_confidence = 0.7

    frequent_itemsets = apriori(data, min_support)
    association_rules = generate_association_rules(frequent_itemsets, min_confidence)

    print_frequent_itemsets(frequent_itemsets)
    print_association_rules(association_rules)


def print_frequent_itemsets(frequent_itemsets):
    # Sắp xếp các itemset theo kích thước và theo thứ tự từ điển
    sorted_itemsets = sorted(frequent_itemsets.items(), key=lambda x: (len(x[0]), sorted(x[0])))

    print("Frequent Itemsets:")
    print("------------------------------------------------------------------------------------")
    print(f"| {'STT':<2} | {'Itemset':<62} | {'Support':<9} |")
    print("|-----+----------------------------------------------------------------+-----------|")

    stt = 1
    for itemset, support in sorted_itemsets:
        # Sắp xếp các phần tử trong itemset theo thứ tự bảng chữ cái
        itemset_list = sorted(itemset)
        itemset_str = ", ".join(itemset_list)  # Ghép các phần tử lại thành chuỗi
        print(f"| {stt:<2}  | [{itemset_str}]{' ' * (60 - len(itemset_str))} | {support*100:>7.2f}%  |")
        stt += 1

    print("------------------------------------------------------------------------------------")


def print_association_rules(association_rules):
    # Sắp xếp luật kết hợp theo kích thước của antecedent và theo thứ tự từ điển
    association_rules.sort(key=lambda rule: (len(rule['antecedent']), sorted(rule['antecedent'])))

    print()
    print("Association Rules:")
    print("--------------------------------------------------------------------------------------")

    stt = 1
    for rule in association_rules:
        # Sắp xếp antecedent và consequent theo thứ tự bảng chữ cái
        antecedent_str = ", ".join(sorted(rule['antecedent']))
        consequent_str = ", ".join(sorted(rule['consequent']))

        # In ra theo định dạng đã sắp xếp
        print(f"| {stt:<2}  | [{antecedent_str}] => [{consequent_str}]{' ' * (60 - len(consequent_str) - len(antecedent_str) - 5)} | {rule['confidence'] * 100:>8.2f}%  |")

        stt += 1

    print("--------------------------------------------------------------------------------------")


# Hàm đọc dữ liệu từ file CSV
def load_dataset_from_csv(filename):
    data = []

    try:
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)

            for row in reader:
                transaction = set()
                for i, val in enumerate(row):
                    if val == 't':
                        transaction.add(headers[i])
                data.append(transaction) #Thêm set đã mua của từng dòng vào mảng data

    except FileNotFoundError:
        print("Cannot read file. Please check your path!")
        return None

    return data



def apriori(data, min_support):
    frequent_itemsets = {}
    k = 1
    itemsets = create_initial_itemsets(data)

    while itemsets:
        support = calculate_support(data, itemsets)
        itemsets = {itemset for itemset, sup in support.items() if sup >= min_support} #Tạo 1 set mới chứa các itemset có support >= min_support sau mỗi lần lặp

        frequent_itemsets.update(support)
        
        # Sinh tập hợp lớn hơn
        itemsets = join_itemsets(itemsets, k)
        k += 1

    return frequent_itemsets


# Tạo tập phổ biến ban đầu (k = 1)
def create_initial_itemsets(data):
    itemsets = set()
    for transaction in data:
        for item in transaction:
            itemsets.add(frozenset([item]))

    return itemsets


# Tính support cho từng tập hợp
def calculate_support(data, itemsets):
    num_transactions = len(data)
    support = {}

    for itemset in itemsets:
        count = sum(1 for transaction in data if itemset.issubset(transaction))
        support[itemset] = count / num_transactions

    return support


# Kết hợp các tập hợp nhỏ hơn để tạo tập lớn hơn
def join_itemsets(itemsets, k):
    new_items = set()
    itemset_list = list(itemsets)

    for i in range(len(itemset_list)):
        for j in range(i + 1, len(itemset_list)):
            item1 = itemset_list[i]
            item2 = itemset_list[j]

            # Kết hợp hai tập hợp
            new_itemset = item1 | item2

            if len(new_itemset) == k + 1:
                subsets = list(combinations(new_itemset, k))  # Dùng thư viện itertools để tìm các tập con dựa trên tổ hợp k
                if all(frozenset(subset) in itemsets for subset in subsets):
                    new_items.add(frozenset(new_itemset))

    return new_items


# Sinh các luật kết hợp từ tập phổ biến
def generate_association_rules(frequent_itemsets, min_confidence):
    association_rules = []

    for itemset, support in frequent_itemsets.items():
        if len(itemset) > 1:
            itemset_list = list(itemset)
            for i in range(1, len(itemset_list)):
                antecedents = generate_combinations(itemset_list, i)
                for antecedent in antecedents:
                    consequent = itemset - antecedent
                    confidence = support / frequent_itemsets[frozenset(antecedent)]
                    if confidence >= min_confidence:
                        rule = {'antecedent': antecedent, 'consequent': consequent, 'confidence': confidence}
                        association_rules.append(rule)

    return association_rules


# Sinh tất cả các tổ hợp (combinations) từ danh sách các phần tử
def generate_combinations(items, k):
    return [frozenset(comb) for comb in combinations(items, k)]

if __name__ == "__main__":
    main()
