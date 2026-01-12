// User status
export var UserStatus;
(function (UserStatus) {
    UserStatus["ACTIVE"] = "ACTIVE";
    UserStatus["DELETED"] = "DELETED";
    UserStatus["BANNED"] = "BANNED";
})(UserStatus || (UserStatus = {}));
// User type
export var UserType;
(function (UserType) {
    UserType["HUMAN"] = "HUMAN";
    UserType["AI"] = "AI";
})(UserType || (UserType = {}));
// Deck status
export var DeckStatus;
(function (DeckStatus) {
    DeckStatus["COMPLETE"] = "COMPLETE";
    DeckStatus["IN_PROGRESS"] = "IN_PROGRESS";
    DeckStatus["DELETED"] = "DELETED";
})(DeckStatus || (DeckStatus = {}));
// Room status
export var RoomStatus;
(function (RoomStatus) {
    RoomStatus["WAITING_FOR_PLAYERS"] = "WAITING_FOR_PLAYERS";
    RoomStatus["DECK_SELECTION"] = "DECK_SELECTION";
    RoomStatus["READY_CHECK"] = "READY_CHECK";
    RoomStatus["STARTING"] = "STARTING";
    RoomStatus["IN_MATCH"] = "IN_MATCH";
    RoomStatus["COMPLETED"] = "COMPLETED";
    RoomStatus["ABORTED"] = "ABORTED";
    RoomStatus["CLOSED"] = "CLOSED";
})(RoomStatus || (RoomStatus = {}));
// Room type
export var RoomType;
(function (RoomType) {
    RoomType["PRIVATE"] = "PRIVATE";
    RoomType["MATCH_MAKING"] = "MATCH_MAKING";
})(RoomType || (RoomType = {}));
// Match result type
export var WinType;
(function (WinType) {
    WinType["WIN"] = "WIN";
    WinType["DRAW"] = "DRAW";
    WinType["ABANDON"] = "ABANDON";
    WinType["FORFEIT"] = "FORFEIT";
    WinType["TIMEOUT"] = "TIMEOUT";
})(WinType || (WinType = {}));
//# sourceMappingURL=index.js.map