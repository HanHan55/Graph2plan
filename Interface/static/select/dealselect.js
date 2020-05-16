var roomvalarr = new Array(14);
for (i = 0; i < 14; i++) {
    roomvalarr[i] = 0;
}
var otherval = 0;
var detailval = 0;
roomvalarr[0] = 1;
// console.log(roomvalarr)
//LivingRoom, MasterRoom, Kitchen,Bathroom,DiningRoom,ChildRoom,StudyRoom
// SecondRoom, GuestRoom, Balcony,Entrance,Storage, Wallin,BedRoom
//0
new Vue({
    el: '#BedRoomVue',
    data() {
        return {
            BedRoomradio1: 'Any',
            BedRoomradio2: '0'
        }
    },
    methods: {
        // {#                    BedRoom#}
        BedRoomnum(val) {
            roomvalarr[13] = val;
            document.getElementById("BedRoomlb").innerHTML = "BedRoom:" + roomvalarr[13];
            if (val == "Any") {
                roomvalarr[13] = 0;
                document.getElementById("BedRoomlb").innerHTML = "BedRoom";

            }
            // console.log(val);
        },
        BedRoomnum_mt(val) {
            roomvalarr[13] = val;
            document.getElementById("BedRoomlb").innerHTML = "BedRoom:" + roomvalarr[13];
            if (val == "0") {
                roomvalarr[13] = 0;
                document.getElementById("BedRoomlb").innerHTML = "BedRoom";

            }

            // console.log(val);
        },
        BedRoom_mt(val) {
            if (val) {
                document.getElementById("BedRoom").style.display = "none";
                document.getElementById("BedRoommt").style.display = "block";

            } else {
                document.getElementById("BedRoom").style.display = "block";
                document.getElementById("BedRoommt").style.display = "none";

            }
            // console.log(val);
        },Done(val) {
                    animateHeight(true);
                            console.log(val);


        }
    }
})
//2
new Vue({
    el: '#BathRoomVue',
    data() {
        return {
            BathRoomradio1: 'Any',
            BathRoomradio2: '0'
        }
    },
    methods: {
        // {#                    BathRoom#}
        BathRoomnum(val) {
            roomvalarr[3] = val;
            document.getElementById("BathRoomlb").innerHTML = "BathRoom:" + roomvalarr[3];
            if (val == "Any") {
                roomvalarr[3] = 0;
                document.getElementById("BathRoomlb").innerHTML = "BathRoom";

            }

            // console.log(val);
        },
        BathRoomnum_mt(val) {
            roomvalarr[3] = val;
            document.getElementById("BathRoomlb").innerHTML = "BathRoom:" + roomvalarr[3];
            if (val == "0") {
                roomvalarr[3] = 0;
                document.getElementById("BathRoomlb").innerHTML = "BathRoom";

            }

            // console.log(val);
        },
        BathRoom_mt(val) {
            if (val) {
                document.getElementById("BathRoom").style.display = "none";
                document.getElementById("BathRoommt").style.display = "block";
            } else {
                document.getElementById("BathRoom").style.display = "block";
                document.getElementById("BathRoommt").style.display = "none";

            }
            // console.log(val);
        },Done2(val) {
                    animateHeight2(true);
                            console.log(val);


        }
    }
})
//1
new Vue({
    el: '#otherVue',
    data() {
        return {
            // LivingRoomradio1: 'Any',
            Kitchenradio1: 'Any',
            Balconyradio1: 'Any',
            DiningRoomradio1: 'Any',
            Entranceradio1: 'Any',
            Storageradio1: 'Any',
            Wallinradio1: 'Any',

            // LivingRoomradio2: '0',
            Kitchenradio2: '0',
            Balconyradio2: '0',
            DiningRoomradio2: '0',
            Entranceradio2: '0',
            Storageradio2: '0',
            Wallinradio2: '0',
        }
    },
    methods: {
        // // {#                    LivingRoom#}
        // LivingRoomnum(val) {
        //     roomvalarr[0] = val;
        //     document.getElementById("LivingRoomlb").innerHTML = "LivingRoom:" + roomvalarr[0];
        //     if (val == "Any") {
        //         roomvalarr[0] = 0;
        //         document.getElementById("LivingRoomlb").innerHTML = "LivingRoom";
        //     }
        //     otherval = countother();
        //     document.getElementById("otherlb").innerHTML =  otherval == 0 ? "Other Room Types" : "Other Room Types:" + otherval;
        //
        //     console.log(val);
        // },
        // LivingRoomnum_mt(val) {
        //     roomvalarr[0] = val;
        //     document.getElementById("LivingRoomlb").innerHTML = "LivingRoom:" + roomvalarr[0];
        //     if (val == "0") {
        //         roomvalarr[0] = 0;
        //         document.getElementById("LivingRoomlb").innerHTML = "LivingRoom";
        //     }
        //     otherval = countother();
        //     document.getElementById("otherlb").innerHTML =  otherval == 0 ? "Other Room Types" : "Other Room Types:" + otherval;
        //
        //     console.log(val);
        // },
        // LivingRoom_mt(val) {
        //     if (val) {
        //         document.getElementById("LivingRoom").style.display = "none";
        //         document.getElementById("LivingRoommt").style.display = "block";
        //     } else {
        //         document.getElementById("LivingRoom").style.display = "block";
        //         document.getElementById("LivingRoommt").style.display = "none";
        //
        //     }
        //     console.log(val);
        // },
        // {#Kitchen#}
        Kitchennum(val) {
            roomvalarr[2] = val;
            document.getElementById("Kitchenlb").innerHTML = "Kitchen:" + roomvalarr[2];
            if (val == "Any") {
                roomvalarr[2] = 0;
                document.getElementById("Kitchenlb").innerHTML = "Kitchen";

            }
            otherval = countother();
            document.getElementById("otherlb").innerHTML =  otherval == 0 ? "Other Room Types" : "Other Room Types:" + otherval;

            // console.log(val);
        },
        Kitchennum_mt(val) {
            roomvalarr[2] = val;
            document.getElementById("Kitchenlb").innerHTML = "Kitchen:" + roomvalarr[2];
            if (val == "0") {
                roomvalarr[2] = 0;
                document.getElementById("Kitchenlb").innerHTML = "Kitchen";

            }
            otherval = countother();
            document.getElementById("otherlb").innerHTML = otherval == 0 ? "Other Room Types" : "Other Room Types:" + otherval;

            // console.log(val);
        },
        Kitchen_mt(val) {
            if (val) {
                document.getElementById("Kitchen").style.display = "none";
                document.getElementById("Kitchenmt").style.display = "block";
            } else {
                document.getElementById("Kitchen").style.display = "block";
                document.getElementById("Kitchenmt").style.display = "none";

            }
            // console.log(val);
        },
        // {#           Balcony         #}
        Balconynum(val) {
            roomvalarr[9] = val;
            document.getElementById("Balconylb").innerHTML = "Balcony:" + roomvalarr[9];
            if (val == "Any") {
                roomvalarr[9] = 0;
                document.getElementById("Balconylb").innerHTML = "Balcony";

            }
            otherval = countother();
            document.getElementById("otherlb").innerHTML = otherval == 0 ? "Other Room Types" : "Other Room Types:" + otherval;

            // console.log(val);
        },
        Balconynum_mt(val) {
            roomvalarr[9] = val;
            document.getElementById("Balconylb").innerHTML = "Balcony:" + roomvalarr[9];
            if (val == "0") {
                roomvalarr[9] = 0;
                document.getElementById("Balconylb").innerHTML = "Balcony";

            }
            otherval = countother();
            document.getElementById("otherlb").innerHTML = otherval == 0 ? "Other Room Types" : "Other Room Types:" + otherval;

            // console.log(val);
        },
        Balcony_mt(val) {
            if (val) {
                document.getElementById("Balcony").style.display = "none";
                document.getElementById("Balconymt").style.display = "block";
            } else {
                document.getElementById("Balcony").style.display = "block";
                document.getElementById("Balconymt").style.display = "none";

            }
            // console.log(val);
        },

        // {#            DiningRoom        #}
        DiningRoomnum(val) {
            roomvalarr[4] = val;
            document.getElementById("DiningRoomlb").innerHTML = "DiningRoom:" + roomvalarr[4];
            if (val == "Any") {
                roomvalarr[4] = 0;
                document.getElementById("DiningRoomlb").innerHTML = "DiningRoom";

            }
            otherval = countother();
            document.getElementById("otherlb").innerHTML =  otherval == 0 ? "Other Room Types" : "Other Room Types:" + otherval;

            // console.log(val);
        },
        DiningRoomnum_mt(val) {
            roomvalarr[4] = val;
            document.getElementById("DiningRoomlb").innerHTML = "DiningRoom:" + roomvalarr[4];
            if (val == "0") {
                roomvalarr[4] = 0;
                document.getElementById("DiningRoomlb").innerHTML = "DiningRoom";

            }
            otherval = countother();
            document.getElementById("otherlb").innerHTML = otherval == 0 ? "Other Room Types" : "Other Room Types:" + otherval;

            // console.log(val);
        },
        DiningRoom_mt(val) {
            if (val) {
                document.getElementById("DiningRoom").style.display = "none";
                document.getElementById("DiningRoommt").style.display = "block";
            } else {
                document.getElementById("DiningRoom").style.display = "block";
                document.getElementById("DiningRoommt").style.display = "none";

            }
            // console.log(val);
        },

        // {#           Entrance         #}
        Entrancenum(val) {
            roomvalarr[10] = val;
            document.getElementById("Entrancelb").innerHTML = "Entrance:" + roomvalarr[10];
            if (val == "Any") {
                roomvalarr[10] = 0;
                document.getElementById("Entrancelb").innerHTML = "Entrance";

            }

            otherval = countother();
            document.getElementById("otherlb").innerHTML =  otherval == 0 ? "Other Room Types" : "Other Room Types:" + otherval;

            // console.log(val);
        },
        Entrancenum_mt(val) {
            roomvalarr[10] = val;
            document.getElementById("Entrancelb").innerHTML = "Entrance:" + roomvalarr[10];
            if (val == "0") {
                roomvalarr[10] = 0;
                document.getElementById("Entrancelb").innerHTML = "Entrance";

            }
            otherval = countother();
            document.getElementById("otherlb").innerHTML = otherval == 0 ? "Other Room Types" : "Other Room Types:" + otherval;

            // console.log(val);
        },
        Entrance_mt(val) {
            if (val) {
                document.getElementById("Entrance").style.display = "none";
                document.getElementById("Entrancemt").style.display = "block";
            } else {
                document.getElementById("Entrance").style.display = "block";
                document.getElementById("Entrancemt").style.display = "none";

            }
            // console.log(val);
        },

        // {#               Storage     #}
        Storagenum(val) {
            roomvalarr[11] = val;
            document.getElementById("Storagelb").innerHTML = "Storage:" + roomvalarr[11];
            if (val == "Any") {
                roomvalarr[11] = 0;
                document.getElementById("Storagelb").innerHTML = "Storage";

            }

            otherval = countother();
            document.getElementById("otherlb").innerHTML = otherval == 0 ? "Other Room Types" : "Other Room Types:" + otherval;

            // console.log(val);
        },
        Storagenum_mt(val) {
            roomvalarr[11] = val;
            document.getElementById("Storagelb").innerHTML = "Storage:" + roomvalarr[11];
            if (val == "0") {
                roomvalarr[11] = 0;
                document.getElementById("Storagelb").innerHTML = "Storage";

            }
            otherval = countother();
            document.getElementById("otherlb").innerHTML = otherval == 0 ? "Other Room Types" : "Other Room Types:" + otherval;

            // console.log(val);
        },
        Storage_mt(val) {
            if (val) {
                document.getElementById("Storage").style.display = "none";
                document.getElementById("Storagemt").style.display = "block";
            } else {
                document.getElementById("Storage").style.display = "block";
                document.getElementById("Storagemt").style.display = "none";

            }
            // console.log(val);
        },

        // {#       Wallin             #}
        Wallinnum(val) {
            roomvalarr[12] = val;
            document.getElementById("Wallinlb").innerHTML = "Wallin:" + roomvalarr[12];
            if (val == "Any") {
                roomvalarr[12] = 0;
                document.getElementById("Wallinlb").innerHTML = "Wallin";

            }
            otherval = countother();
            document.getElementById("otherlb").innerHTML = otherval == 0 ? "Other Room Types" : "Other Room Types:" + otherval;

            // console.log(val);
        },
        Wallinnum_mt(val) {
            roomvalarr[12] = val;
            document.getElementById("Wallinlb").innerHTML = "Wallin:" + roomvalarr[12];
            if (val == "0") {
                roomvalarr[12] = 0;
                document.getElementById("Wallinlb").innerHTML = "Wallin";

            }
            otherval = countother();
            document.getElementById("otherlb").innerHTML = otherval == 0 ? "Other Room Types" : "Other Room Types:" + otherval;

            // console.log(val);
        },
        Wallin_mt(val) {
            if (val) {
                document.getElementById("Wallin").style.display = "none";
                document.getElementById("Wallinmt").style.display = "block";
            } else {
                document.getElementById("Wallin").style.display = "block";
                document.getElementById("Wallinmt").style.display = "none";

            }
            // console.log(val);
        },Done1(val) {
                    animateHeight1(true);
                            console.log(val);


        }

    }
})
//3
new Vue({
    el: '#detailVue',
    data() {
        return {
            MasterRoomradio1: 'Any',
            GuestRoomradio1: 'Any',
            SecondRoomradio1: 'Any',
            ChildRoomradio1: 'Any',
            StudyRoomradio1: 'Any',

            MasterRoomradio2: '0',
            GuestRoomradio2: '0',
            SecondRoomradio2: '0',
            ChildRoomradio2: '0',
            StudyRoomradio2: '0',
        }
    },
    methods: {
        // {#                    MasterRoom#}
        MasterRoomnum(val) {
            roomvalarr[1] = val;
            document.getElementById("MasterRoomlb").innerHTML = "MasterRoom:" + roomvalarr[1];

            if (val == "Any") {
                roomvalarr[1] = 0;
                document.getElementById("MasterRoomlb").innerHTML = "MasterRoom";

            }
            detailval = countdeailted();
            document.getElementById("detailedlb").innerHTML = detailval == 0 ? "Detailed Bedroom Types" : "Detailed Bedroom Types:" + detailval;

            // console.log(val);
        },
        MasterRoomnum_mt(val) {
            roomvalarr[1] = val;
            document.getElementById("MasterRoomlb").innerHTML = "MasterRoom:" + roomvalarr[1];


            if (val == "0") {
                roomvalarr[1] = 0;
                document.getElementById("MasterRoomlb").innerHTML = "MasterRoom";


            }
            detailval = countdeailted();
            document.getElementById("detailedlb").innerHTML = detailval == 0 ? "Detailed Bedroom Types" : "Detailed Bedroom Types:" + detailval;

            // console.log(val);
        },
        MasterRoom_mt(val) {
            if (val) {
                document.getElementById("MasterRoom").style.display = "none";
                document.getElementById("MasterRoommt").style.display = "block";
            } else {
                document.getElementById("MasterRoom").style.display = "block";
                document.getElementById("MasterRoommt").style.display = "none";

            }
            // console.log(val);
        },

        // {#                    GuestRoom#}
        GuestRoomnum(val) {
            roomvalarr[8] = val;
            document.getElementById("GuestRoomlb").innerHTML = "GuestRoom:" + roomvalarr[8];

            if (val == "Any") {
                roomvalarr[8] = 0;
                document.getElementById("GuestRoomlb").innerHTML = "GuestRoom";


            }
            detailval = countdeailted();

            document.getElementById("detailedlb").innerHTML = detailval == 0 ? "Detailed Bedroom Types" : "Detailed Bedroom Types:" + detailval;

            // console.log(val);
        },
        GuestRoomnum_mt(val) {
            roomvalarr[8] = val;
            document.getElementById("GuestRoomlb").innerHTML = "GuestRoom:" + roomvalarr[8];

            if (val == "0") {
                roomvalarr[8] = 0;
                document.getElementById("GuestRoomlb").innerHTML = "GuestRoom";


            }
            detailval = countdeailted();

            document.getElementById("detailedlb").innerHTML = detailval == 0 ? "Detailed Bedroom Types" : "Detailed Bedroom Types:" + detailval;

            // console.log(val);
        },
        GuestRoom_mt(val) {
            if (val) {
                document.getElementById("GuestRoom").style.display = "none";
                document.getElementById("GuestRoommt").style.display = "block";
            } else {
                document.getElementById("GuestRoom").style.display = "block";
                document.getElementById("GuestRoommt").style.display = "none";

            }
            // console.log(detailval);

            // console.log(val);
        },
        // {#                    SecondRoom#}
        SecondRoomnum(val) {
            roomvalarr[7] = val;
            document.getElementById("SecondRoomlb").innerHTML = "SecondRoom:" + roomvalarr[7];

            if (val == "Any") {
                roomvalarr[7] = 0;
                document.getElementById("SecondRoomlb").innerHTML = "SecondRoom";


            }
            detailval = countdeailted();

            document.getElementById("detailedlb").innerHTML = detailval == 0 ? "Detailed Bedroom Types" : "Detailed Bedroom Types:" + detailval;

            // console.log(val);
        },
        SecondRoomnum_mt(val) {
            roomvalarr[7] = val;
            document.getElementById("SecondRoomlb").innerHTML = "SecondRoom:" + roomvalarr[7];

            if (val == "0") {
                roomvalarr[7] = 0;
                document.getElementById("SecondRoomlb").innerHTML = "SecondRoom";


            }
            detailval = countdeailted();

            document.getElementById("detailedlb").innerHTML = detailval == 0 ? "Detailed Bedroom Types" : "Detailed Bedroom Types:" + detailval;

            // console.log(val);
        },
        SecondRoom_mt(val) {
            if (val) {
                document.getElementById("SecondRoom").style.display = "none";
                document.getElementById("SecondRoommt").style.display = "block";
            } else {
                document.getElementById("SecondRoom").style.display = "block";
                document.getElementById("SecondRoommt").style.display = "none";

            }
            // console.log(val);
        },
        // {#                    ChildRoom#}
        ChildRoomnum(val) {
            roomvalarr[5] = val;
            document.getElementById("ChildRoomlb").innerHTML = "ChildRoom:" + roomvalarr[5];

            if (val == "Any") {
                roomvalarr[5] = 0;
                document.getElementById("ChildRoomlb").innerHTML = "ChildRoom";


            }
            detailval = countdeailted();

            document.getElementById("detailedlb").innerHTML = detailval == 0 ? "Detailed Bedroom Types" : "Detailed Bedroom Types:" + detailval;
            // console.log(val);
        },
        ChildRoomnum_mt(val) {
            roomvalarr[5] = val;
            document.getElementById("ChildRoomlb").innerHTML = "ChildRoom:" + roomvalarr[5];

            if (val == "0") {
                roomvalarr[5] = 0;
                document.getElementById("ChildRoomlb").innerHTML = "ChildRoom";


            }
            detailval = countdeailted();

            document.getElementById("detailedlb").innerHTML = detailval == 0 ? "Detailed Bedroom Types" : "Detailed Bedroom Types:" + detailval;

            // console.log(val);
        },
        ChildRoom_mt(val) {
            if (val) {
                document.getElementById("ChildRoom").style.display = "none";
                document.getElementById("ChildRoommt").style.display = "block";
            } else {
                document.getElementById("ChildRoom").style.display = "block";
                document.getElementById("ChildRoommt").style.display = "none";

            }
            // console.log(val);
        },
        // {#                    StudyRoom#}
        StudyRoomnum(val) {
            roomvalarr[6] = val;
            document.getElementById("StudyRoomlb").innerHTML = "StudyRoom:" + roomvalarr[6];
            if (val == "Any") {
                roomvalarr[6] = 0;
                document.getElementById("StudyRoomlb").innerHTML = "StudyRoom";


            }
            detailval = countdeailted();

            document.getElementById("detailedlb").innerHTML = detailval == 0 ? "Detailed Bedroom Types" : "Detailed Bedroom Types:" + detailval;

            // console.log(val);
        },
        StudyRoomnum_mt(val) {
            roomvalarr[6] = val;
            document.getElementById("StudyRoomlb").innerHTML = "StudyRoom:" + roomvalarr[6];

            if (val == "0") {
                roomvalarr[6] = 0;
                document.getElementById("StudyRoomlb").innerHTML = "StudyRoom";


            }
            detailval = countdeailted();

            document.getElementById("detailedlb").innerHTML = detailval == 0 ? "Detailed Bedroom Types" : "Detailed Bedroom Types:" + detailval;

            // console.log(val);
        },
        StudyRoom_mt(val) {
            if (val) {
                document.getElementById("StudyRoom").style.display = "none";
                document.getElementById("StudyRoommt").style.display = "block";
            } else {
                document.getElementById("StudyRoom").style.display = "block";
                document.getElementById("StudyRoommt").style.display = "none";

            }
            // console.log(val);
        },Done3(val) {
                    animateHeight3(true);
                            console.log(val);


        }

    },
})
new Vue({
    el: '#addVue',
    data() {
        return {
        }
    },
    methods: {

    },
})

function initVue() {
    var vue1=new Vue({ el: '#BedRoomVue' });
    // vue1.BedRoomradio1= 'Any';
    //         vue1.BedRoomradio2= '0';
        vue1.BedRoomradio1="Any";
        // document.getElementById("detailVue").assign(this.$data, this.$options.data());
        // document.getElementById("otherVue").assign(this.$data, this.$options.data());
        // document.getElementById("BathRoomVue").assign(this.$data, this.$options.data());

}

function countdeailted() {
    var count = 0;
    if (roomvalarr[1] != 0) count = count + 1;
    if (roomvalarr[5] != 0) count = count + 1;
    if (roomvalarr[6] != 0) count = count + 1;
    if (roomvalarr[7] != 0) count = count + 1;
    if (roomvalarr[8] != 0) count = count + 1;
    return count
}

function countother() {

    var count = 0;
    if (roomvalarr[2] != 0) count = count + 1;
    if (roomvalarr[9] != 0) count = count + 1;
    if (roomvalarr[4] != 0) count = count + 1;
    if (roomvalarr[10] != 0) count = count + 1;
    if (roomvalarr[11] != 0) count = count + 1;
    if (roomvalarr[12] != 0) count = count + 1;
// console.log(roomvalarr);
    return count
}

function Num() {
    var detailindx = [1, 5, 6, 7, 8];
    var temp = 0;
    var bedroomsum = 0;
    //detail :no,bedroom:exact
    if (roomvalarr[13].length == 1 || roomvalarr[13] == 0)
        if (roomvalarr[1].length == 2 || roomvalarr[5].length == 2 || roomvalarr[6].length == 2 || roomvalarr[7].length == 2 || roomvalarr[8].length == 2) {
            for (i = 0; i < 5; i++) {
                if (roomvalarr[detailindx[i]].length == 2) {
                    temp = temp + parseInt(roomvalarr[detailindx[i]].split("+")[0]) + 1;
                } else
                    temp = temp + roomvalarr[detailindx[i]];
            }
            // console.log("temp", temp);
            if (roomvalarr[13] < temp)
                if (temp < 6)
                    roomvalarr[13] = temp;
                else roomvalarr[13] = 5;
            roomvalarr[13] = String(roomvalarr[13]) + "+";
        }
    //detail :exact,bedroom:exact
    if (roomvalarr[13].length == 1 || roomvalarr[13] == 0) {
        if ((roomvalarr[1].length == 1 || roomvalarr[1] == 0) &&(roomvalarr[5].length == 1 || roomvalarr[5] == 0) && (roomvalarr[6].length == 1 || roomvalarr[6] == 0)  && (roomvalarr[7].length == 1 || roomvalarr[7] == 0)  && (roomvalarr[8].length == 1 || roomvalarr[8] == 0) ) {
            bedroomsum = parseInt(roomvalarr[1] )+parseInt( roomvalarr[5]) + parseInt(roomvalarr[6]) + parseInt(roomvalarr[7] )+ parseInt(roomvalarr[8]);
            // console.log("bedroomsum", bedroomsum);

            if (roomvalarr[13] < bedroomsum)
                if (bedroomsum < 6)
                    roomvalarr[13] = bedroomsum;
                else roomvalarr[13] = 5;
            roomvalarr[13] = String(roomvalarr[13]);

        }

    }
if(roomvalarr[13]=="0")roomvalarr[13]=0;
    // console.log("roomvalarr[13]", roomvalarr[13]);
    var roomactarr = new Array(14);
    var roomexaarr = new Array(14);
    var roomnumarr = new Array(14);
roomactarr[0]=1;roomexaarr[0] = 1;roomnumarr[0] = 1;
    for (i = 0; i < 14; i++) {
        if (roomvalarr[i] == 0) {
            roomactarr[i] = 0;
            roomexaarr[i] = 0;
            roomnumarr[i] = 0
        }
        if (roomvalarr[i].length == 2) {
            roomactarr[i] = 1;
            roomexaarr[i] = 0;
            roomnumarr[i] = roomvalarr[i].split("+")[0];
        }
        if (roomvalarr[i].length == 1) {
            roomactarr[i] = 1;
            roomexaarr[i] = 1;
            roomnumarr[i] = roomvalarr[i];
        }

    }
    return {
        roomactarr: roomactarr,
        roomexaarr: roomexaarr,
        roomnumarr: roomnumarr,

    }
}

