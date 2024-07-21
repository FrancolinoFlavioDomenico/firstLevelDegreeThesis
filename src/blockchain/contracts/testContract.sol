// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;

contract testContract {
  string public testString;      

   function setTestString(string memory _testString) public {         
      testString = _testString;     
   } 
 
   function getTestString() public view returns(string memory) {         
      return testString;     
   } 

    function getSimpleString() public pure returns(string memory) {
        return "fixedTestString";
    }

}
