#include <iostream>
using namespace std;

void insertionSort(int arr[], int n){

    int i, key, j;

    for(int i = 1; i < n; i++){
        comp = arr[i];
        j = i-1;

        cout << "New Iteration";
        cout << "comp: " << comp;
        cout << j;

        while(j >= 0 && arr[j] > comp){

            arr[j+1] = arr[j];
            j = j-1;
            printArray(arr);
        }

        arr[j+1] = comp;
        cout << "After Iteration: ";
        printArray(arr);
    }
}

void mergeSort(int array[], int l, int r){
    
    if (l < r){
        int m = (l+r)/2;
        mergeSort(array, l, m);
        mergeSort(array, m+1, r);
        //merge(A)
    }
}

void printArray(int arr[]){
    for(int i = 0; i < n; i++){
        cout << arr[i];
    }
    cout << "\n";
}

int main(){
    int arr[] = {12,11,13,5,6};
    int n = sizeof(arr);
    insertionSort(arr, n);

    //int l = 0;
    //int r = n-1;

    //mergeSort(arr, 1, r);
}
