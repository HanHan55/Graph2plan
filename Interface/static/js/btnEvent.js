window.onload = function () {
    let cookies = document.cookie.split(";");
    for (let i = 0; i < cookies.length; i++) {
        let cookie = cookies[i];
        let eqPos = cookie.indexOf("=");
        let name = eqPos > -1 ? cookie.substr(0, eqPos) : cookie;
        document.cookie = name + "=;expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/";
    }
    var IMAGE_PATH = "../static/images/";
    document.cookie = "RoomIndex=0";
    //初始化数
    var BedNum = 0, MasterNum = 0, SecondNum = 0, GuestNum = 0, StudyNum = 0, ChildNum = 0, LivingNum = 0;
    var KitchenNum = 0, BalconyNum = 0, BathNum = 0, DiningNum = 0, StorageNum = 0, EntranceNum = 0, Wall_inNum = 0;
    // document.getElementById("BedRoom").onclick = function () {
    //     var curInd = changeRoomIndexCookie();
    //     ++BedNum;
    //     document.getElementById("BedRoom_val").innerHTML = BedNum;
    //     newElement("BedRoom", BedNum, curInd);
    // }
    // document.getElementById("MasterRoom").onclick = function () {
    //     var curInd = changeRoomIndexCookie();
    //     ++MasterNum;
    //     ++BedNum;
    //     newElement("MasterRoom", MasterNum, curInd);
    //     document.getElementById("MasterRoom_val").innerHTML = MasterNum;
    //     document.getElementById("BedRoom_val").innerHTML = BedNum;
    // }
    // document.getElementById("SecondRoom").onclick = function () {
    //     var curInd = changeRoomIndexCookie();
    //     ++SecondNum;
    //     ++BedNum;
    //     newElement("SecondRoom", SecondNum, curInd);
    //     document.getElementById("SecondRoom_val").innerHTML = SecondNum;
    //     document.getElementById("BedRoom_val").innerHTML = BedNum;
    // }
    // document.getElementById("GuestRoom").onclick = function () {
    //     var curInd = changeRoomIndexCookie();
    //     ++GuestNum;
    //     ++BedNum;
    //     newElement("GuestRoom", GuestNum, curInd);
    //     document.getElementById("GuestRoom_val").innerHTML = GuestNum;
    //     document.getElementById("BedRoom_val").innerHTML = BedNum;
    // }
    // document.getElementById("StudyRoom").onclick = function () {
    //     var curInd = changeRoomIndexCookie();
    //     ++StudyNum;
    //     ++BedNum;
    //     newElement("StudyRoom", StudyNum, curInd);
    //     document.getElementById("StudyRoom_val").innerHTML = StudyNum;
    //     document.getElementById("BedRoom_val").innerHTML = BedNum;
    // }
    // document.getElementById("ChildRoom").onclick = function () {
    //     var curInd = changeRoomIndexCookie();
    //     ++ChildNum;
    //     ++BedNum;
    //     newElement("ChildRoom", ChildNum, curInd);
    //     document.getElementById("ChildRoom_val").innerHTML = ChildNum;
    //     document.getElementById("BedRoom_val").innerHTML = BedNum;
    // }
    // document.getElementById("LivingRoom").onclick = function () {
    //     var curInd = changeRoomIndexCookie();
    //     ++LivingNum;
    //     newElement("LivingRoom", LivingNum, curInd);
    //     document.getElementById("LivingRoom_val").innerHTML = LivingNum;
    // }
    // document.getElementById("Kitchen").onclick = function () {
    //     var curInd = changeRoomIndexCookie();
    //     ++KitchenNum;
    //     newElement("Kitchen", KitchenNum, curInd);
    //     document.getElementById("Kitchen_val").innerHTML = KitchenNum;
    // }
    // document.getElementById("Balcony").onclick = function () {
    //     var curInd = changeRoomIndexCookie();
    //     ++BalconyNum;
    //     newElement("Balcony", BalconyNum, curInd);
    //     document.getElementById("Balcony_val").innerHTML = BalconyNum;
    // }
    // document.getElementById("BathRoom").onclick = function () {
    //     var curInd = changeRoomIndexCookie();
    //     ++BathNum;
    //     newElement("BathRoom", BathNum, curInd);
    //     document.getElementById("BathRoom_val").innerHTML = BathNum;
    // }
    // document.getElementById("DiningRoom").onclick = function () {
    //     var curInd = changeRoomIndexCookie();
    //     ++DiningNum;
    //     newElement("DiningRoom", DiningNum, curInd);
    //     document.getElementById("DiningRoom_val").innerHTML = DiningNum;
    // }
    // document.getElementById("Storage").onclick = function () {
    //     var curInd = changeRoomIndexCookie();
    //     ++StorageNum;
    //     newElement("Storage", StorageNum, curInd);
    //     document.getElementById("Storage_val").innerHTML = StorageNum;
    // }
    // document.getElementById("Entrance").onclick = function () {
    //     var curInd = changeRoomIndexCookie();
    //     ++EntranceNum;
    //     newElement("Entrance", EntranceNum, curInd);
    //     document.getElementById("Entrance_val").innerHTML = EntranceNum;
    // }
    // document.getElementById("Wallin").onclick = function () {
    //     var curInd = changeRoomIndexCookie();
    //     ++Wall_inNum;
    //     newElement("Wallin", Wall_inNum, curInd);
    //     document.getElementById("Wallin_val").innerHTML = Wall_inNum;
    // }

    var animationDiv = document.getElementById('animationDiv');
    var addButton = document.getElementById('add');
    var removeSVG = '<?xml version="1.0" encoding="utf-8"?><svg version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 22 22" style="enable-background:new 0 0 22 22;" xml:space="preserve"><g><g><path class="fill" d="M16.1,3.6h-1.9V3.3c0-1.3-1-2.3-2.3-2.3h-1.7C8.9,1,7.8,2,7.8,3.3v0.2H5.9c-1.3,0-2.3,1-2.3,2.3v1.3c0,0.5,0.4,0.9,0.9,1v10.5c0,1.3,1,2.3,2.3,2.3h8.5c1.3,0,2.3-1,2.3-2.3V8.2c0.5-0.1,0.9-0.5,0.9-1V5.9C18.4,4.6,17.4,3.6,16.1,3.6z M9.1,3.3c0-0.6,0.5-1.1,1.1-1.1h1.7c0.6,0,1.1,0.5,1.1,1.1v0.2H9.1V3.3z M16.3,18.7c0,0.6-0.5,1.1-1.1,1.1H6.7c-0.6,0-1.1-0.5-1.1-1.1V8.2h10.6L16.3,18.7L16.3,18.7z M17.2,7H4.8V5.9c0-0.6,0.5-1.1,1.1-1.1h10.2c0.6,0,1.1,0.5,1.1,1.1V7z"/></g><g><g><path class="fill" d="M11,18c-0.4,0-0.6-0.3-0.6-0.6v-6.8c0-0.4,0.3-0.6,0.6-0.6s0.6,0.3,0.6,0.6v6.8C11.6,17.7,11.4,18,11,18z"/></g><g><path class="fill" d="M8,18c-0.4,0-0.6-0.3-0.6-0.6v-6.8C7.4,10.2,7.7,10,8,10c0.4,0,0.6,0.3,0.6,0.6v6.8C8.7,17.7,8.4,18,8,18z"/></g><g><path class="fill" d="M14,18c-0.4,0-0.6-0.3-0.6-0.6v-6.8c0-0.4,0.3-0.6,0.6-0.6c0.4,0,0.6,0.3,0.6,0.6v6.8C14.6,17.7,14.3,18,14,18z"/></g></g></g></svg>';
    var checkSVG = ' <?xml version="1.0" encoding="utf-8"?><svg version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 22 22" style="enable-background:new 0 0 22 22;" xml:space="preserve"><circle  cx=11 cy=11 class="noFill" stroke="rgb(47, 167, 77)" r="10" width="23" height="23"/><g><path class="fill" d="M9.7,14.4L9.7,14.4c-0.2,0-0.4-0.1-0.5-0.2l-2.7-2.7c-0.3-0.3-0.3-0.8,0-1.1s0.8-0.3,1.1,0l2.1,2.1l4.8-4.8c0.3-0.3,0.8-0.3,1.1,0s0.3,0.8,0,1.1l-5.3,5.3C10.1,14.3,9.9,14.4,9.7,14.4z"/></g></svg>';


    completedList = document.getElementById('completed');
    todoList = document.getElementById('todo');


// Core Functions

    function addItem(roomname) {
        if (itemTextBox.value) {
            newElement(itemTextBox.value);
            itemTextBox.value = "";
        }

        document.getElementById('add').classList.toggle("rotate");
        addAnimation();

    }


    function fade(element) {
        element.classList.remove('invisible');

    }


    function checkItem() {
        checkButton = this;
        checkAnimateDiv = this.children[1];
        console.log(checkAnimateDiv);
        checkElement = this.parentElement.parentElement;


        if (!(checkElement.classList.contains('checked'))) {
            selectRoomType(checkElement.innerText, this.id);
            checkElement.classList.add('checked');
            checkAnimation(checkAnimateDiv);

            //checkElement.classList.add('invisible');

            setTimeout(function () {
                completedList.appendChild(checkElement);

            }, 100);

            setTimeout(function () {
                DOMCleaner(checkAnimateDiv);

            }, 500);
        } else {
            checkElement.classList.remove('checked');
            deleteAnimation(checkElement);
            todoList.insertBefore(checkElement, todoList.firstChild);

            var points = d3.select("body").selectAll("circle");

            var curInd = this.id.split("_")[1];

            points.each(function (d, i) {
                var tmpInd = this.id.split("_")[1];
                if (tmpInd == curInd) {
                    d3.select(this).remove();
                }
            });
        }


    }


    function DOMCleaner(item) {
        while (item.firstChild) {
            //item.removeChild(item.firstChild);
            item.firstChild.remove();
        }
    }

    function removeItem(name, deleteButton) {
        deleteElement = deleteButton.parentElement.parentElement;
        deleteElement.classList.add('deleted');
        deleteAnimation(deleteElement);
        switch (name) {
            case "BedRoom":
                var a = --BedNum;
                break;
            case "MasterRoom":
                var a = --MasterNum;
                --BedNum;
                break;
            case "SecondRoom":
                var a = --SecondNum;
                --BedNum;
                break;
            case "GuestRoom":
                var a = --GuestNum;
                --BedNum;
                break;
            case "StudyRoom":
                var a = --StudyNum;
                --BedNum;
                break;
            case "ChildRoom":
                var a = --ChildNum;
                --BedNum;
                break;
            case "Kitchen":
                var a = --KitchenNum;
                break;
            case "DiningRoom":
                var a = --DiningNum;
                break;
            case "BathRoom":
                var a = --BathNum;
                break;
            case "Balcony":
                var a = --BalconyNum;
                break;
            case "Entrance":
                var a = --EntranceNum;
                break;
            case "Wallin":
                var a = --Wall_inNum;
                break;
            case "Storage":
                var a = --StorageNum;
                break;
            case "LivingRoom":
                var a = --LivingNum;
                break;

            default:
                break
        }
        var id = name + "_val";
        document.getElementById(id).innerHTML = a;
        document.getElementById("BedRoom_val").innerHTML = BedNum;

        setTimeout(function () {
            deleteElement.remove();
        }, 500);


    }


    function newElement(name, value, curInd) {

        var item_image = document.createElement('img');
        item_image.src = IMAGE_PATH + name + ".png";
        item_image.classList.add('showrmitem');

        var item = document.createElement('li');

        item.innerHTML = name;
        item.appendChild(item_image);
        item.classList.add('invisible');


        var buttonsDiv = document.createElement('div');
        buttonsDiv.classList.add('buttons');

        var deleteButton = document.createElement('button');
        deleteButton.classList.add('deleteButton');
        deleteButton.id = "Delete_" + curInd;
        deleteButton.innerHTML = removeSVG;
        deleteButton.onclick = function () {
            if (d3.select("body").selectAll("circle").length > 0) {
                var points = d3.select("body").selectAll("circle");
                var curInd = this.id.split("_")[1];

                points.each(function (d, i) {
                    var tmpInd = this.id.split("_")[1];
                    if (tmpInd == curInd) {
                        d3.select(this).remove();
                    }
                });
            }


            removeItem(name, deleteButton);
        }


        var checkButton = document.createElement('button');
        checkButton.classList.add('checkButton');
        checkButton.innerHTML = checkSVG;
        checkButton.id = "Check_" + curInd;

        var checkAnimateDiv = document.createElement('div');
        checkAnimateDiv.id = "checkAnimateDiv";
        checkButton.appendChild(checkAnimateDiv);

        checkButton.addEventListener('click', checkItem);

        buttonsDiv.appendChild(deleteButton);
        buttonsDiv.appendChild(checkButton);

        item.appendChild(buttonsDiv);


        //todoList.prepend(item);
        todoList.insertBefore(item, todoList.firstChild);
        setTimeout(function () {
            fade(item);
        }, 100);


    }


// Visual


    /* Add Button Animation */
    function addAnimation() {
        const Burst1 = new mojs.Burst({
            parent: animationDiv,
            top: '50%',
            left: '50%',
            radius: {0: 80},
            count: 8,
            children: {
                shape: 'circle',
                fill: {'red': 'blue'},
                strokeWidth: 1,
                duration: 600,
                stroke: {'red': 'blue'}
            }
        });


        const Burst2 = new mojs.Burst({
            parent: animationDiv,
            top: '50%',
            left: '50%',
            radius: {0: 100},
            count: 4,
            children: {
                shape: 'rect',
                fill: 'white',
                strokeWidth: 1,
                duration: 300,
                stroke: 'white'
            }
        });


        const circle1 = new mojs.Shape({
            radius: {0: 40},
            parent: animationDiv,
            fill: 'none',
            stroke: 'white',
            strokeWidth: 15,
            duration: 300,
            opacity: {1: 0}
        });

        const circle2 = new mojs.Shape({
            radius: {0: 50},
            parent: animationDiv,
            fill: 'none',
            stroke: 'red',
            strokeWidth: 5,
            duration: 400,
            opacity: {1: 0}
        });


        const circle3 = new mojs.Shape({
            radius: {0: 60},
            parent: animationDiv,
            fill: 'none',
            stroke: 'blue',
            strokeWidth: 5,
            duration: 500,
            opacity: {1: 0}
        });

        const circle4 = new mojs.Shape({
            radius: {0: 70},
            parent: animationDiv,
            fill: 'white',

            stroke: 'white',
            strokeWidth: 5,
            duration: 600,
            opacity: {1: 0}
        });

        const timeline = new mojs.Timeline({

            repeat: 0
        }).add(circle4, circle1, circle2, circle3, Burst1, Burst2);

        timeline.play();
    }

    /* Delete item animation */

    function checkAnimation(checkItem) {
        const circle1 = new mojs.Shape({
            radius: {0: 1000},
            parent: checkItem,
            fill: '#7bef28',
            stroke: 'white',
            strokeWidth: 10,
            duration: 500,
            opacity: {1: 0}
        });

        const circle2 = new mojs.Shape({
            radius: {0: 200},
            parent: checkItem,
            fill: 'none',
            stroke: 'white',
            strokeWidth: 30,
            duration: 300,
            opacity: {1.7: 0}
        });

        const circle3 = new mojs.Shape({
            radius: {0: 400},
            parent: checkItem,
            fill: 'none',
            stroke: '#230e5780',
            strokeWidth: 10,
            duration: 400,
            opacity: {1: 0}

        });


        const timelineX = new mojs.Timeline({
            repeat: 0,

        }).add(circle1, circle2, circle3);

        timelineX.play();


    }




    function changeRoomIndexCookie() {
        var arr, reg = new RegExp("(^| )RoomIndex=([^;]*)(;|$)");

        var curIndex = -1;
        if (arr = document.cookie.match(reg))
            curIndex = arr[2];

        document.cookie = "RoomIndex=" + (parseInt(curIndex) + 1);

        return curIndex;
    }
}
